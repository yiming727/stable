import numpy as np
import os
import math
import re
from ir_color import ppbyIron


class keii_data_load():

    def __init__(self):
        pass
    
    #------------------------------------------------------------#
    #   一、IRV格式读取, f= open(video_path,"rb")
    #------------------------------------------------------------# 
    # 读IRV的帧数据的长宽
    def Read_IRVFrame_wh(self,f):
        #分辨率
        #有两种存宽高的方法
        f.seek(256) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
        w=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
        w= int(w, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
        h=f.read(2).hex()
        h= int(h, base=16)
        if 0 in [w,h]:
            f.seek(256+2)
            w=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
            w= int(w, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
            f.seek(256+6)
            h=f.read(2).hex()
            h= int(h, base=16)
        f.close
        return w,h

    # 读IRV的总帧数
    def Read_TotalFrame_IRV(self,f):     
        # #分辨率
        # f.seek(256) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
        # w=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
        # w= int(w, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
        # h=f.read(2).hex()
        # h= int(h, base=16)
        # if 0 in [w,h]:
        #     f.seek(256+2)
        #     w=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
        #     w= int(w, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
        #     f.seek(256+6)
        #     h=f.read(2).hex()
        #     h= int(h, base=16)

        # f.seek(0,2) 
        # size=f.tell()

        # Total_num = int((size-32*1024)/(w*h*2+1024))-1
        # f.close

        # 读取文件尾部数据
        f.seek(-512, 2)  # 跳转到文件末尾前 512 字节的位置
        last_data = f.read(512)
        # 提取尾部标记（最后 3 字节）
        EndRe = last_data[-3:]
        # 检查尾部标记是否为 'ABC'
        flag = EndRe == b'ABC'
        if flag:
            # print("尾部标记匹配: 是 'ABC'")
            # 提取总帧数
            total_frame = int.from_bytes(last_data[:8], byteorder='little')  # 直接解析为整数
        else:
            # print(f"尾部标记不匹配: 实际值为 {EndRe}")
            # 如果尾部标记不符合预期，重新解码总帧数
            frame_bytes = last_data[:8]
            total_frame = int.from_bytes(frame_bytes, byteorder='little')  # 直接解析为整数
            i64T = total_frame & 0xffffffff  # 提取低 32 位
            total_frame = (
                ((i64T >> 24) & 0xff) +        # 高字节
                ((i64T >> 16) & 0xff) * 256 +  # 次高字节
                ((i64T >> 8) & 0xff) * 65536 + # 次低字节
                (i64T & 0xff) * 16777216       # 最低字节
            )
            # print(f"重新解码后的总帧数: {total_frame}")
        return total_frame

    def modify_totalFrame_IRV(self,last_data, new_framenum):
        """

        参数:
            last_data (bytes): 文件末尾的 512 字节数据。
            new_framenum (int): 新的总帧数。

        返回:
            bytes: 更新后的 512 字节数据。

        异常:
            ValueError: 如果输入数据长度不正确。
        """
        # 检查输入数据长度是否为 512 字节
        if len(last_data) != 512:
            raise ValueError("last_data 的长度必须为 512 字节")

        # 提取尾部标记（最后 3 字节）
        EndRe = last_data[-3:]

        # 检查尾部标记是否为 'ABC'
        if EndRe != b'ABC':
            # 提取原 last_data 的前 8 字节并转换为小端序整数
            original_total_frame = int.from_bytes(last_data[:8], byteorder='little')
            i64T_1 = (
                ((new_framenum >> 24) & 0xff) +         # 高字节
                ((new_framenum >> 16) & 0xff) * 256 +   # 次高字节
                ((new_framenum >> 8) & 0xff) * 65536 +  # 次低字节
                (new_framenum & 0xff) * 16777216       # 最低字节
            )
            # 获取 high 32 位部分（总帧数的高 32 位）
            high_32_bits = (original_total_frame >> 32) & 0xffffffff

            # 将恢复的 i64T_1 和高 32 位部分组合成 total_frame_2
            total_frame_2 = (high_32_bits << 32) | i64T_1
            new_frame_bytes = total_frame_2.to_bytes(8, byteorder='little')
        else:
            # 将新的总帧数转换为 8 字节的小端字节数据
            new_frame_bytes = new_framenum.to_bytes(8, byteorder='little')

        # 替换 last_data 的前 8 字节为新的总帧数
        new_last_data = new_frame_bytes + last_data[8:]
        # 检查输入数据长度是否为 512 字节
        if len(new_last_data) != 512:
            raise ValueError("new_last_data 的长度必须为 512 字节")
        return new_last_data

    # 读IRV的帧数据
    def Open_Frame_IRV(self,f,frame,width=None,hight=None):
        if ((width == None) and (hight == None)):
            # # 将文件指针移动到文件末尾
            # f.seek(0, 2)  # 2 表示从文件末尾开始偏移
            # # 获取文件大小
            # file_size = f.tell()
            #分辨率
            f.seek(256) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
            width=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
            width= int(width, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
            hight=f.read(2).hex()
            hight= int(hight, base=16)
            if 0 in [width,hight]:
                f.seek(256+2)
                width=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
                width= int(width, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
                f.seek(256+6)
                hight=f.read(2).hex()
                hight= int(hight, base=16)

        #frame  = 10   #irv：32*1024 视频的头，(640*480*2+1024) 一帧的长度
        f.seek(32*1024+(frame)*(hight*width*2+1024),0) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
        ccc=f.read(hight*width*2) #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
        img= np.frombuffer(ccc, dtype='uint16')#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
        if img.size == hight * width:
            img = img.reshape((hight, width))
        else:
            return None 
        return img 

    # 读IRV的帧数据
    def Open_Frame_IRV2(self,frame_data,frame,width,height):
         # 计算当前帧的起始位置
        frame_start =  frame * (height * width * 2 + 1024)
        frame_end = frame_start + height * width * 2
        # 提取当前帧的数据
        ccc = frame_data[frame_start:frame_end]
        # 将字节数据转换为 NumPy 数组
        img = np.frombuffer(ccc, dtype='uint16')
        # 检查图像大小是否正确
        if img.size == height * width:
            img = img.reshape((height, width))  # 转换为二维数组
        else:
            return None  # 如果大小不匹配，返回 None

        return img
        
    # 读IRV的帧数据
    def Open_keyframe_IRV(self,f,frame):
        #分辨率
        f.seek(256) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
        width=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
        width= int(width, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
        hight=f.read(2).hex()
        hight= int(hight, base=16)
        if 0 in [width,hight]:
            f.seek(256+2)
            width=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
            width= int(width, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
            f.seek(256+6)
            hight=f.read(2).hex()
            hight= int(hight, base=16)
        # 读镜头类型和测温档位、辐射率
        f.seek(32*1024+(frame-1)*(hight*width*2+1024)+(1)*(hight*width*2+408),0) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
        try:
            # 读取一个字节
            byte_data = f.read(1)
            # 检查是否为空
            if not byte_data:
                print("文件已结束或无数据可读")
                return None
            # 将字节数据转换为十六进制字符串并解析为整数
            keyframe_point = int(byte_data.hex(), base=16)
            return keyframe_point
        except ValueError as e:
            # 捕获其他可能的异常
            print(f"解析错误: {e}")
            return None


    # 读IRV的温度标定矩阵(辐射率修正)
    def Read_AD2Templist_IRV(self,f,frame=2):
        try:
            #镜头类型
            f.seek(256) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
            width=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
            width= int(width, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
            hight=f.read(2).hex()
            hight= int(hight, base=16)
            if 0 in [width,hight]:
                f.seek(256+2)
                width=f.read(2).hex() #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
                width= int(width, base=16)#read的数值以bytes的类型保存，通过np.frombuffer方法还原成类型为uint16的ndarray，这种方式还原出来的ndarray是只读的。
                f.seek(256+6)
                hight=f.read(2).hex()
                hight= int(hight, base=16)
            # 读镜头类型和测温档位、辐射率
            f.seek(32*1024+(frame-1)*(hight*width*2+1024)+(1)*(hight*width*2+175),0) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
            lensNum = int(f.read(1).hex(),base=16)
            f.seek(32*1024+(frame-1)*(hight*width*2+1024)+(1)*(hight*width*2+92),0) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
            iCeWenRange = int(f.read(1).hex(),base=16)
            f.seek(32*1024+(frame-1)*(hight*width*2+1024)+(1)*(hight*width*2+69),0) # 第一个代表需要移动偏移的字节数,0 代表从文件开头开始算起
            EmsDec =  int(f.read(1).hex(),base=16)/100
            #读温度标定
            TempDes_id  = 4352+(lensNum*3+iCeWenRange)*3+1; #得到温度保存地址
            f.seek(TempDes_id) 
            TempBDes1   = int(f.read(2).hex(),base=16)
            f.seek(TempBDes1+2) 
            num   = int(f.read(2).hex(),base=16)
            f.seek(TempBDes1+4) 
            temp = []
            for i in range(0,num,1):
                tempi   = int(f.read(2).hex(),base=16)
                if tempi>8000:
                    tempi=tempi-65536
                temp.append(tempi/10)

            #读AD标定
            ADDes_id=  4096+(lensNum*3+iCeWenRange)*3+1; #得到温度保存地址  +1
            f.seek(ADDes_id) 
            ADBDes   = int(f.read(2).hex(),base=16)
            f.seek(ADBDes+4) 
            AD = []
            for i in range(0,num,1):
                ADi   = int(f.read(2).hex(),base=16)/EmsDec
                AD.append(ADi)

            Temperature_list =np.zeros((num,2),dtype = np.int32)
            Temperature_list[:,0] = temp
            Temperature_list[:,1] = AD
            # f.close
        except ValueError as e:
            # 捕获其他可能的异常
            print(f"解析错误: {e}")
            return None

        return Temperature_list

    #------------------------------------------------------------#
    #   二、keii_jpg 格式读取
    #------------------------------------------------------------# 
    # 读keii格式jpg
    def Open_Keii_Jpg(self,data):
        #---------------------------------------------------#
        #  判断数据格式
        #---------------------------------------------------#
        if isinstance(data, bytes):  
            fileDataArray =  data
        elif os.path.isfile(data) : 
            path = data
            with open(path, "rb") as f:
                buf_size = 2*1024 * 1024  # 1 MB
                fileDataArray = f.read(buf_size)
        else:
            return None 

        buf = fileDataArray
        len = (f.tell())  
        offset = len - 11
        f.seek(offset,0)
        suffix = f.read(11)
        if((suffix !=  b'KEII-IR-JPG') & (suffix !=  b'KEII-PP-JPG')) :
            return None
        f.close()

        # 文件版本号
        offset -= 2
        # 融合图片数
        offset -= 1
        picCount = buf[offset]
        # picCount = 1
        # IMA数据文件
        offset -= picCount * 14
        # 测温分析信息长度
        offset -= 2
        analysisLen = (buf[offset] << 8) + buf[offset + 1]
        # 测温分析信息
        offset -= analysisLen
        # 文本备注信息长度
        offset -= 2
        txtRemarkLen = (buf[offset] << 8) + buf[offset + 1]
        # 文本备注信息
        offset -= txtRemarkLen
        # 音频注释长度
        offset -= 4
        audioRemarkLen = (buf[offset] << 24) + (buf[offset + 1] << 16) + (buf[offset + 2] << 8) + buf[offset + 3]
        # 音频注释
        offset -= audioRemarkLen
        offset -= 92
        #红外点阵高度
        offset -= 2
        hight = (buf[offset] << 8) + buf[offset + 1]
        # #红外点阵宽度
        offset -= 2
        width = (buf[offset] << 8) + buf[offset + 1]
        # width = 640
        # hight = 480
        imaLen = 32 + hight * width * 2 + 1024
        offset -= imaLen

        f = open(path,"rb")
        f.seek(offset+32,0)
        ccc=f.read(hight * width * 2) #python在读取文件的时候是根据光标位置来读取的。读一行以后光标位置到了下一行。再来个read又到了下一行。
        u16img= np.frombuffer(ccc, dtype='uint16')
        if u16img.size >0:
            u16img = u16img.reshape((hight, width))
        else:
            u16img= np.zeros((hight,width),dtype=np.int16)
        return u16img 

    # 读keii格式jpg和温度矩阵(辐射率修正)   
    def Open_Keii_Jpg_Temp(self,data):
        #---------------------------------------------------#
        #  判断数据格式
        #---------------------------------------------------#
        if isinstance(data, bytes):  
            fileDataArray =  data
        elif os.path.isfile(data) : 
            path = data
            with open(path, "rb") as f:
                buf_size = 2*1024 * 1024  # 1 MB
                fileDataArray = f.read(buf_size)
        else:
            return None,None,None
        #---------------------------------------------------#
        #  判断是否为keil格式数据
        #---------------------------------------------------#
        buf = fileDataArray
        length= len(buf)
        offset = length - 11
        suffix = buf[offset:offset+11]
        if((suffix !=  b'KEII-IR-JPG') & (suffix !=  b'KEII-PP-JPG')) :
            return None,None,None
        # 文件版本号
        offset -= 2
        # 融合图片数
        offset -= 1
        # picCount = buf[offset]
        picCount = 1
        # IMA数据文件
        offset -= picCount * 14
        # 测温分析信息长度
        offset -= 2
        analysisLen = (buf[offset] << 8) + buf[offset + 1]
        offset -= analysisLen
        # 文本备注信息长度
        offset -= 2
        # txtRemarkLen = (buf[offset] << 8) + buf[offset + 1]
        txtRemarkLen = 0
        # 文本备注信息
        offset -= txtRemarkLen
        # 音频注释长度
        offset -= 4
        # audioRemarkLen = (buf[offset] << 24) + (buf[offset + 1] << 16) + (buf[offset + 2] << 8) + buf[offset + 3]
        audioRemarkLen = 0
        # 音频注释
        offset -= audioRemarkLen
        offset -= 92
        #红外点阵高度
        offset -= 2
        hight = (buf[offset] << 8) + buf[offset + 1]
        # #红外点阵宽度
        offset -= 2
        width = (buf[offset] << 8) + buf[offset + 1]

        # IMA图像数据
        ima_conf =  1024
        offset -= ima_conf
        lensNum = buf[offset+175]
        iCeWenRange= buf[offset+92]
        EmsDec =  buf[offset+69]/100
        if EmsDec == 0 :
            EmsDec = 1

        imaLen = hight * width * 2 
        offset -= imaLen
        ccc = buf[offset:(offset+hight * width * 2)]
        u16img= np.frombuffer(ccc, dtype='uint16')
        if u16img.size >0:
            u16img = np.uint16(u16img.reshape((hight, width)))
        else:
            u16img= None

        # IMA-32K头
        offset -= 32*1024
        useful_bit = buf[offset+331]
        #读温度标定
        TempDes_id = 4352+(lensNum*3+iCeWenRange)*3+1; #得到温度保存地址
        TempBDes1   = (buf[offset+TempDes_id] << 8) + buf[offset+TempDes_id+1]
        if offset+TempBDes1 > length-20:
            return None,None,None

        num = (buf[offset+TempBDes1+2] << 8) + buf[offset+TempBDes1+2+1]
        if num >50:
            return None,None,None
        #温度轴数据
        temp = []
        for i in range(0,num,1):
            tempi  = (buf[offset+TempBDes1+4+2*i] << 8) + buf[offset+TempBDes1+4+2*i+1]
            if tempi>8000:
                tempi=tempi-65536
            temp.append(tempi/10)
        #AD值轴数据,标定表里已经是有效ad值
        ADDes_id =  4096+(lensNum*3+iCeWenRange)*3+1; #得到温度保存地址
        ADBDes = (buf[offset+ADDes_id] << 8) + buf[offset+ADDes_id+1]
        AD = []
        for i in range(0,num,1):
            ADi   = ((buf[offset+ADBDes+4+2*i] << 8) + buf[offset+ADBDes+4+2*i+1])/EmsDec
            AD.append(ADi)
        Temperature_list =np.zeros((num,2),dtype = np.int32)
        Temperature_list[:,0] = temp
        Temperature_list[:,1] = AD
        # 有效ad值判断
        u16img = self.useful_AD(u16img,Temperature_list)
        #去边缘黑边
        # u16img=u16img[2:-2,2:-2] 
        #获取图像的温度
        Temperature_IMG = self.AD2Temperature(u16img,Temperature_list)

        return u16img,Temperature_IMG,Temperature_list
    
    #------------------------------------------------------------#
    #   三、简单工具函数
    #------------------------------------------------------------# 
    #有效ad值判断，12位存储时将16位帧数据转12位
    def useful_AD(self,u16img,Temperature_list):
        admax = np.max(Temperature_list[:,1])
        if admax < 30000:
            temp =  np.ones_like(u16img)* 0x3fff  #非制冷型热像仪
            u16img = u16img & temp
        return u16img
    
    #根据温度标定矩阵和16位帧数据，获取帧温度矩阵
    def AD2Temperature(self,u16img,Temperature_list):
        Temperature_IMG = np.zeros_like(u16img).astype(np.float64) # list比np快
        admax = np.max(Temperature_list[:,1])
        if admax < 30000:
            temp =  np.ones_like(u16img)* 0x3fff  #非制冷型热像仪
            u16img = u16img & temp

        for j in range(0,Temperature_list.shape[0]):
            if j == 0:
                idx = ((u16img) <= Temperature_list[j][1])
            else:
                idx = ((u16img) > Temperature_list[j-1][1]) & ((u16img) <= Temperature_list[j][1])
                
            Temperature_IMG[idx] = Temperature_list[j-1][0]  +10*(u16img[idx]-Temperature_list[j-1][1])/(Temperature_list[j][1]-Temperature_list[j-1][1])
        return Temperature_IMG

    #根据颜色表将灰度转RGB显示
    def ima_to_rgb(self,in_img,RGBlist):

        h=in_img.shape[0] 
        w=in_img.shape[1]
        T=in_img.reshape(1,h*w)
        
        R=RGBlist[T,0]
        G=RGBlist[T,1]
        B=RGBlist[T,2]
        R=R.reshape(h,w)
        G=G.reshape(h,w)
        B=B.reshape(h,w)

        out_img = np.zeros((h,w,3), 'uint8')
        out_img[:,:,0]=R
        out_img[:,:,1]=G
        out_img[:,:,2]=B

        return  out_img

    #线性拉伸
    def imadjust(self,in_img,low,high):
        out_img=in_img* ((low<=in_img)&(in_img<= high))+high* (in_img > high)+low*(in_img < low)
        out_img=(out_img-low)/(high-low)*255
        out_img=np.uint8(out_img)
        return out_img

    
    # 解码测温分析信息
    def calculate_slope_and_intercept(self,point1, point2):
        """
        根据两点坐标计算线段的方向、斜率和截距。
        参数:
        - point1: 第一个点的坐标 (x1, y1)
        - point2: 第二个点的坐标 (x2, y2)
        返回:
        - direction: 方向，0 表示水平，1 表示垂直
        - a: 斜率
        - b: 截距
        """
        (x1, y1), (x2, y2) = point1, point2
        delta_x = x2 - x1
        delta_y = y2 - y1
        if abs(delta_y) > abs(delta_x):  # 更接近垂直
            direction = 1
            a = delta_x / delta_y if delta_y != 0 else 0  # 避免除以零
            b = x1 - a * y1
        else:  # 更接近水平
            direction = 0
            a = delta_y / delta_x if delta_x != 0 else 0  # 避免除以零
            b = y1 - a * x1
        return direction, a, b
        
    
    def distance(self,p1, p2):
        """计算两点之间的距离"""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_linesInfo(self,line_items):
        # 存储线的坐标、斜率和截距信息
        lines_ab = []
        lines_point = []
        # 处理每个以 "L:" 开头的项
        for line_item in line_items:
            # 去掉 'L:' 前缀并提取点的信息
            points_info = line_item[2:]  # 去掉 'L:' 前缀
            points = points_info.split(";")[:2]  # 以 ';' 分割不同点，取前两个点
            # 将每个坐标对转换为列表
            coordinates = [list(map(float, point.split(','))) for point in points if point]
            if len(coordinates) == 2:  # 确保有两个点
                point1, point2 = coordinates[0],coordinates[1]
                direction,a,b = self.calculate_slope_and_intercept(point1, point2)
                if direction == 1 :
                    if point1[1] > point2[1]:
                        coordinates[0],coordinates[1] = coordinates[1],coordinates[0]
                if direction == 0 :
                    if point1[0] > point2[0]:
                        coordinates[0],coordinates[1] = coordinates[1],coordinates[0]
                # 存储线的坐标、斜率和截距
                lines_ab.append([direction,a,b])
                lines_point.append(coordinates)

        return lines_ab,lines_point

    # 计算 det_line_point 到直线 y = ax + b 的距离
    def point_to_line_distance(self,point, a,b,direction):
        x = point[0]
        y = point[1]
        if direction ==0:
            # ax - y + b = 0
            distance = abs(a * x - y + b) / np.sqrt(a ** 2 + 1)
        else:
            # x = ay + b
            # x - ay + b = 0
            distance = abs(x - a * y  - b) / np.sqrt(a ** 2 + 1)
        return distance

    def merge_lines(self,lines_ab, lines_point):
        """
        合并相似的线段，返回新的线段及其斜率和截距信息。

        参数:
        - lines_ab: 包含线段斜率和截距信息的列表，每个元素为 [direction, a, b]
        - lines_point: 包含线段起点和终点坐标的列表，每个元素为 [[x1, y1], [x2, y2]]

        返回:
        - new_lines_ab: 合并后的线段的斜率和截距信息
        - new_lines_point: 合并后的线段的起点和终点坐标
        """
        new_lines_ab = []
        new_lines_point = []
        processed_indices = set()  # 记录已经处理过的线段索引

        for i in range(len(lines_ab)):
            if i in processed_indices:
                continue
            line1_ab = lines_ab[i]
            line1_points = lines_point[i]
            merged = False
            for j in range(i + 1, len(lines_ab)):
                if j in processed_indices:
                    continue
                line2_ab = lines_ab[j]
                line2_points = lines_point[j]
                 # 计算 line1 两个点到直线line2_ab的距离
                upper_distance = point_to_line_distance(line1_points[0], line2_ab[1], line2_ab[2], line2_ab[0])
                lower_distance = point_to_line_distance(line1_points[1], line2_ab[1], line2_ab[2], line2_ab[0])
                distance = (upper_distance + lower_distance)
                # 斜率和截距的相似性检查
                if distance < 15:

                    # 合并线段，新的线段由 line1 的起点和 line2 的终点组成
                    direction = line1_ab[0]
                    if direction == 1 :
                        start_idx = np.argmax([np.array(line1_points)[:,1],np.array(line2_points)[:,1]])
                        if start_idx == 1:
                            line1_points,line2_points = line2_points,line1_points 
                    if direction == 0 :
                        start_idx = np.argmax([np.array(line1_points)[:,0],np.array(line2_points)[:,0]])
                        if start_idx == 1:
                            line1_points,line2_points = line2_points,line1_points 
                        
                    new_line_points = [line1_points[0], line2_points[1]]
                    new_lines_point.append(new_line_points)
                    # 重新计算新线段的斜率和截距
                    direction, a, b = self.calculate_slope_and_intercept(new_line_points[0], new_line_points[1])
                    new_lines_ab.append([direction, a, b])
                    # 将两个线段的索引标记为已处理
                    processed_indices.add(i)
                    processed_indices.add(j)
                    merged = True
                    break

            # 如果没有合并，则保留原来的线段
            if not merged:
                new_lines_ab.append(line1_ab)
                new_lines_point.append(line1_points)
                processed_indices.add(i)
        return new_lines_ab, new_lines_point

    # 读keii格式jpg和温度矩阵(辐射率修正)   
    def Open_Keii_Jpg_lineInfo(self,data):
        lines_ab = []
        lines_point = []
        #---------------------------------------------------#
        #  判断数据格式
        #---------------------------------------------------#
        if isinstance(data, bytes):  
            fileDataArray =  data
        elif os.path.isfile(data) : 
            path = data
            with open(path, "rb") as f:
                buf_size = 2*1024 * 1024  # 1 MB
                fileDataArray = f.read(buf_size)
        else:
            return lines_ab,lines_point
        #---------------------------------------------------#
        #  判断是否为keil格式数据
        #---------------------------------------------------#
        buf = fileDataArray
        length= len(buf)
        offset = length - 11
        suffix = buf[offset:offset+11]
        if((suffix !=  b'KEII-IR-JPG') & (suffix !=  b'KEII-PP-JPG')) :
            return lines_ab,lines_point
        # 文件版本号
        offset -= 2
        # 融合图片数
        offset -= 1
        # picCount = buf[offset]
        picCount = 1
        # IMA数据文件
        offset -= picCount * 14
        # 测温分析信息长度
        offset -= 2
        analysisLen = (buf[offset] << 8) + buf[offset + 1]
        # 提取分析信息
        analysis_info = buf[offset - analysisLen:offset].decode('utf-8')
        # 提取线的分析信息
        parts = re.split(r'-(?!\d)', analysis_info)     # r'-(?!\d)'表示'-'后面不能跟数字。
        line_items = [item for item in parts if item.startswith('L:')]
        if line_items:
            # 提取中心线信息
            lines_ab,lines_point =self.get_linesInfo(line_items)
            # 合并斜率截距相似的线
            lines_ab, lines_point = self.merge_lines(lines_ab, lines_point)
        return lines_ab, lines_point


