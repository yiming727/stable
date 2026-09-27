import numpy as np
import cv2
from keii_sdk.ir_color import ppbyIron

class img_trans():
    def __init__(self):
        pass
    
    def detect_crop_imhance(self,img,detect_result,img_info=None,high_th=0.0065):
        #筛选TC和FH的结果
        result_tcfh = list(filter(lambda x: (x['label'] == 'FH')or(x['label'] == 'TC'), detect_result))
        # 结果变量初始化
        fuse_image = np.copy(img)
        for  i  in  result_tcfh:
            contoursPolygon=i["contoursPolygon"]      
            # 裁切目标   
            cropimage,fuse_image= self.proMaskCrop(fuse_image, contoursPolygon,img_info,high_th)
        return fuse_image

    #---------------------------------------------------#
    #   读取图片
    #---------------------------------------------------#
    def keii_read_u16img(self,u16img):
        #---------------------------------------------------#
        #  细节层
        #---------------------------------------------------#
        # 导向滤波划分细节
        guide_img =(np.copy(u16img)).astype("float32")
        eps =0.2 * 255 * 255       # ε最好不要超过0.1，ε较小时，细节层将由一些小的细节组成，例如背景噪音和微小的纹理结构。较大的ε将忽略某些纹理，并特别注意梯度大的边缘。
        radius = 2                  # 引导滤波使用的卷积核半径，滤波窗口变大平滑效果增加，图像更加模糊
        # base0 = cv2.ximgproc.guidedFilter(guide=guide_img, src=guide_img, radius=radius, eps=eps)
        base0,mean_a,mean_b = self.guideFilter(guide_img, guide_img, radius, eps)  # 效果不如官方
        # 细节层
        detail= u16img - base0.astype("float32")
        #---------------------------------------------------#
        #  融合层
        #---------------------------------------------------#
        # 细节增强
        fuse_u16 = (u16img + 0.4*detail)
        fuse_u16 =  (fuse_u16 - np.min(fuse_u16)).astype("uint32")
        # 直方图整理，Lowratio越大低灰度拉的越开
        u8img = self.Hist16to8_new(fuse_u16,Lowratio = 8)
        return u8img

    #------------------------------#
    #   一、气体常用
    #------------------------------# 
    # 直方图整理压缩：16位分批转8位算法， 低灰度优先、分批通过
    def Hist16to8_new(self, img ,Lowratio = 8):
        total = img.shape[0] * img.shape[1]
        T=img.ravel()
        total = T.shape[0]
        value_min = np.min(T)
        value_max = np.max(T)
        # 计算直方图  
        hist16 = np.bincount(T, minlength=65536)
        if hist16[0]>5000:
            hist16[0] = 0
        # 计算累积直方图  
        cumulative_hist = np.cumsum(hist16) / total  
        # 灰度集中区域 5%-90%
        threshold_10 = 0.05  
        threshold_90 = 0.90  
        index_10 = np.where(cumulative_hist >= threshold_10)[0][0]  
        index_90 = np.where(cumulative_hist >= threshold_90)[0][0]
        origin_span = index_90 - index_10
        # 自适应直方图整理范围
        span = origin_span * 2
        if span >3000:
            span = 3000
        #16位到8位映射表
        his16to8 = [0 for element in range(value_max+1)]
        PassAll = 0 #所有通过的16位灰度数量
        Waitnum = 0 #同一车厢正在等待的16位灰度
        Order = 1 #当前排到的8位车厢号
        for i in range(np.max([value_min,1]),value_max+1):
            histV = hist16[i]
            if(histV == 0):
                continue
            #低灰度补偿系数，防止低灰度获得过多座位
            # Lowratio = 18
            if(Order < Lowratio):
                Lowratio  = Lowratio - Order
            else:
                Lowratio = 1  

            his16to8[i] = Order         # 当前16位灰度映射到当前8位灰度
            PassAll = PassAll + histV   # 所有通过的16位灰度数量
            Waitnum = Waitnum + histV   # 当前8位灰度级包含16位灰度级数量
            # 动态分配每个8位灰度级可以容纳的16位灰度级数量，Lowratio越大Pass_threshold越小
            Pass_threshold = (total-PassAll)/(span-Order)/Lowratio  
            #是否满足通过阈值 pass_threshold
            if Waitnum > Pass_threshold : 
                Waitnum = 0            #等待人数清空
                Order = Order +1       #启用下一个8位灰度级，Order+1号      
            if Order >= span-1:
                Order = span-1            #多余的全部都去最后一节车厢
        
        # 将输入16位灰度img根据映射表his16to8映射成新的8位灰度
        his16to8 = np.array(his16to8)
        new_img = np.zeros((total), dtype=np.uint16)  
        # 定义阈值变量
        threshold1 = int(210/255 * span)  # 第一个阈值：210 * radio
        threshold2 = int(96/255 * span)   # 第二个阈值：96 * radio
        threshold3 = int(40/255 * span)   # 第三个阈值：40 * radio
        if (Order<threshold1 & Order>=threshold2):
            new_img = his16to8[T] + 10
        elif(Order<threshold2 & Order>=threshold3):
            new_img = his16to8[T]*1.2 + 10
        elif(Order<threshold3):
            new_img = his16to8[T]*1.5 + 10
        else:
            new_img = his16to8[T]
        new_img = np.array(new_img,dtype=(np.float32)).reshape(img.shape[0],img.shape[1])
        new_img = np.uint8(cv2.normalize(new_img,None,0,255,cv2.NORM_MINMAX))
        return new_img

    def Hist_fuse16to8(self,u16_channel_data):
        hist_fuse = []
        for c in range(3):  # 通道维度
            if c == 0:  # 增强中等灰度区域
                channel_mean = np.mean(u16_channel_data)
                bounds = (channel_mean-5000, channel_mean+5000)
            elif c == 1:  # 压缩高灰度，拉伸低灰度
                # 保留更多低灰度细节（5%~60%区间）
                lower = np.percentile(u16_channel_data, 5)
                upper = np.percentile(u16_channel_data, 60)
                bounds = (lower-2000,upper+500)
            elif c == 2:  # 压缩低灰度，拉伸高灰度
                # 保留更多高灰度细节（40%~95%区间）
                lower = np.percentile(u16_channel_data, 40)
                upper = np.percentile(u16_channel_data, 95)
                bounds = (lower-500,upper+2000)
            channel_data = np.clip(u16_channel_data, bounds[0], bounds[1])
            # 统一进行归一化处理
            equalized_channel = cv2.normalize(channel_data, None, 0, 1, cv2.NORM_MINMAX)
            hist_fuse.append(equalized_channel)
        u8_img = ((hist_fuse[0].astype(np.float32)+hist_fuse[1]+hist_fuse[2])/3*255).astype(np.uint8)
        return u8_img

        
   # 直方图16位分批转8位算法， 低灰度优先、分批通过(需优化)
    def Hist16to8(self,img):
        total = img.shape[0] * img.shape[1]
        T=img.ravel()
        value_min = np.min(img)
        value_max = np.max(img)
        hist16 = np.bincount(T, minlength=65536)
        
        # 16位到8位映射表
        his16to8 = [0 for element in range(65536)]
        PassAll = 0 #所有通过的16位灰度数量
        Waitnum = 0 #同一车厢正在等待的16位灰度
        Order = 1 #当前排到的8位车厢号

        # 需要优化，value_min到value_max的范围太大了，循环速度太慢
        # 应考虑 从每个8位灰度级容纳的16位灰度级数量 grayLevelsPer8Bit 角度做灰度级映射，将范围减少到0-255
        for i in range(value_min,value_max+1):
            #低灰度补偿系数，防止低灰度获得过多座位
            Lowratio = 8
            if(Order <Lowratio):
                Lowratio  = Lowratio - Order
            else:
                Lowratio = 1  
            histV = hist16[i]
            if(histV == 0):
                continue
            
            Waitnum = Waitnum + histV #同一车厢正在等待的16位灰度
            #计算每个8位灰度级应分配的16位像素数量
            Pass_threshold = (total-PassAll)/(256-Order)/Lowratio  #下一车厢可以容纳的16位灰度数量
            
            PassAll = PassAll + histV #所有通过的16位灰度数量

            #是否满足通过阈值 pass_threshold
            ratio = np.floor(Waitnum / Pass_threshold)  
            if(ratio>0):
                his16to8[i] = Order    #去Order号车厢
                Waitnum = 0            #等待人数清空
                Order = Order +1       #启用下一个车厢，Order+1号
            else:
                his16to8[i] = Order    #去Order号车厢
            if Order >= 255:
                Order = 255            #多余的全部都去最后一节车厢

        # #将输入16位灰度img根据映射表his16to8映射成新的8位灰度
        new_img = [0 for element in range(total)]

        for i in range(0,total):
        #根据最后填满的8位灰度数量，进一步调整
            if (Order<210 & Order>=96):
                new_img[i] = his16to8[T[i]] + 10
            elif(Order<96 & Order>=40):
                new_img[i] = his16to8[T[i]]*1.2 + 10
            elif(Order<40):
                new_img[i] = his16to8[T[i]]*1.5 + 10
            else:
                new_img[i] = his16to8[T[i]]

        new_img=np.array(new_img,dtype=(np.float32)).reshape(img.shape[0],img.shape[1])
        new_img = np.uint8(cv2.normalize(new_img,None,0,255,cv2.NORM_MINMAX))
        return new_img
    
    # 直方图16位分批转8位算法， 低灰度优先、分批通过
    def Hist16to8_v1(self,img):
        # start_time = time.time()
        h = img.shape[0]
        w = img.shape[1]
        total = h*w
        T=img.ravel()
        value_min = np.min(img)
        # value_max = np.max(img)

        # 计算216位图像直方图
        hist16 = np.bincount(T, minlength=65536)
        
        # 计算16位到8位映射表
        his16to8 = [0 for element in range(65536)]
        PassAll = 0 #所有通过的16位灰度数量
        u16Waitnum = 0 #同一车厢正在等待的16位灰度

        # 计算每个8位灰度级应分配的16位像素数量，将循环范围减少到0-255
        u16Order = value_min #当前排到的16位车厢号
        for u8Order in range(0,256):
            # 计算每个8位灰度级应分配的16位像素数量
            Pass_threshold = (total-PassAll)/(256-u8Order)# 每个8位灰度级应分配的16位像素数量 = 剩下的像素数/剩下的8位灰度级
            # 根据数量阈值将16位灰度级映射到8位灰度级
            u16OrderStart = u16Order+1
            if u8Order >=255:
                his16to8[u16OrderStart:] = [255 for element in range(65536-u16OrderStart+1)]    #所有通过的16位灰度级映射到当前8位灰度级
            else:
                while((u16Waitnum <= Pass_threshold)&(u16Order<65535)):
                    # 当前16位灰度级像素数量
                    histNum = hist16[u16Order]
                    # 把当前16位灰度级映射到当前的8位灰度
                    if (u8Order<210 & u8Order>=96):
                        his16to8[u16Order+1] = u8Order + 10
                    elif(u8Order<96 & u8Order>=40):
                        his16to8[u16Order+1] = u8Order*1.2 + 10
                    elif(u8Order<40):
                        his16to8[u16Order+1] = u8Order*1.5 + 10
                    else:
                        his16to8[u16Order+1] = u8Order
                    # 当前的8位灰度容纳的16位灰度级像素数量
                    u16Waitnum = u16Waitnum + histNum
                    # 下一个16位灰度级
                    u16Order =  u16Order + 1
                #所有通过的16位灰度数量
                PassAll = PassAll + u16Waitnum 
                #初始化
                u16Waitnum = 0   

        # 将输入16位灰度img根据映射表his16to8映射成新的8位灰度
        # 使用 NumPy 的矢量化操作进行映射，这比循环快得多
        new_img = np.zeros((h, w), dtype=np.uint8)  
        his16to8 = np.array(his16to8)
        new_img[:] = his16to8[T].reshape(img.shape)

        return new_img

    # 多帧分段拉伸增强
    def HSMP(self,inputA, inputB, valueA, valueB, valueBv):
        height, width = inputA.shape
        lengthWH = height * width

        inputA = inputA.ravel() #将inputA展平成一个一维数组
        inputB = inputB.ravel()

        SumA = np.uint64(inputA.sum()) #求和
        SumB = np.uint64(inputB.sum())
        ValueSubAB = (inputA - inputB.astype(np.int32)) #差帧

        #最大/小值
        MaxV = np.max(ValueSubAB)
        MinV = np.min(ValueSubAB)

        #平均值
        avgA = SumA / lengthWH
        avgB = SumB / lengthWH
        avgA_B = avgA - avgB.astype(np.float32)

        #valueB=50，valueBv=70，valueA = 0
        ValueB_A = valueB - valueA      #50
        ValueBV_A = valueBv - valueA    #70
        Value127_BV = 127 - valueBv     #57
        ValueMax_B = np.abs(MaxV - avgA_B - valueB)
        ValueMin_B = np.abs(MinV - avgA_B - valueB)
        if (ValueB_A == 0):	    ValueB_A = 1
        if (ValueMax_B == 0):	ValueMax_B = 1
        if (ValueMin_B == 0):	ValueMin_B = 1


        result = np.empty(lengthWH, dtype=np.int32)
        
        Value = ValueSubAB - avgA_B.astype(np.float32)
        Vabs = abs(Value)
        Vabs_sqrt = np.sqrt(Vabs)

        idx = (Value > 0) & (Vabs < valueA)
        result[idx] = np.floor(Vabs_sqrt + 127)[idx]
        idx = (Value > 0) & (Vabs >= valueA) & (Vabs < valueB)
        result[idx] = (np.floor(valueA + ((Vabs - valueA) * 100 / (ValueB_A))  * ValueBV_A / 100) + 127)[idx]
        idx = (Value > 0) & (Vabs >= valueB)
        result[idx] = np.floor(valueBv + ((Vabs- valueB) * 100 / ValueMax_B) * Value127_BV / 100 + 127)[idx]
        idx = (Value <= 0) & (Vabs < valueA)
        result[idx] = -np.floor(Vabs_sqrt + 127)[idx]
        idx  = (Value <= 0) & (Vabs >= valueA) & (Vabs < valueB)
        result[idx] = -np.floor( valueA + ((Vabs - valueA) * 100 / ValueB_A) * ValueBV_A / 100 + 127 )[idx]
        idx = (Value <= 0) & (Vabs >= valueB)
        result[idx] = -np.floor(valueBv + ((Vabs - valueB) * 100 / ValueMin_B) * Value127_BV / 100 + 127)[idx]

        output = result.reshape((height, width)).astype(np.uint8)
        return output


    

    # 导向滤波
    # 导向滤波
    def guideFilter(self,I, g, radius, eps):
        winSize = int(radius*2 +1)
        """I:导入的图像， g:引导图像"""
        mean_I = cv2.boxFilter(I, ddepth=-1, ksize=(winSize,winSize), normalize=1)  # I的均值平滑
        mean_g = cv2.boxFilter(g, ddepth=-1, ksize=(winSize,winSize), normalize=1)  # g的均值平滑

        mean_gg = cv2.boxFilter(g * g, ddepth=-1, ksize=(winSize,winSize), normalize=1)  # I*I的均值平滑
        mean_Ig = cv2.boxFilter(I * g, ddepth=-1, ksize=(winSize,winSize), normalize=1)  # I*g的均值平滑

        var_g = mean_gg - mean_g * mean_g  # 方差
        cov_Ig = mean_Ig - mean_I * mean_g  # 协方差

        a = cov_Ig / (var_g + eps)  # 相关因子a
        b = mean_I - a * mean_g  # 相关因子b

        mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(winSize,winSize), normalize=1)  # 对a进行均值平滑
        mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(winSize,winSize), normalize=1)  # 对b进行均值平滑

        out = mean_a * g + mean_b

        return out,mean_a,mean_b
    

    def HS_ProAB(self, inputA, inputB, update_degree=1000):

        height, width = inputA.shape[:2]

        inputA = np.asarray(inputA, dtype=np.int32).reshape(-1)
        inputB = np.asarray(inputB, dtype=np.int32).reshape(-1)
        
        difAB = inputB - inputA
        AbsAB = np.abs(difAB)
        
        MaxAB = np.max(difAB)
        MinAB = np.min(difAB)
        avgAB = np.mean(difAB)
        
        AB_Max = max(np.abs(MaxAB - avgAB), 1.0)
        AB_Min = max(np.abs(MinAB - avgAB), 1.0)
        
        bg_noise = 2
        output = np.full_like(difAB, 127, dtype=np.int32)
        
        # Positive differences with significant change
        pos_mask = (difAB > 0) & (AbsAB >= bg_noise)
        output[pos_mask] = ((10 + (((AbsAB[pos_mask] - bg_noise) * update_degree) / AB_Max)) + 127).astype(int)
        
        # Negative differences with significant change
        neg_mask = (difAB <= 0) & (AbsAB >= bg_noise)
        output[neg_mask] = (-(10 + ((AbsAB[neg_mask] - bg_noise) * update_degree / AB_Min)) + 127).astype(int)
        
        # Clip values
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output.reshape((height, width))

    # 原图细节增强
    # GF& DDE ，基于引导滤波的数字细节增强(DDE)算法，当引导滤波器应用在气体红外图像上时，较小的窗口尺寸r和较小的ε将能得到满意的细节层
    def GF_DDE(self,imgA):
        
        # 导向滤波划分细节
        imgA =(imgA).astype("float32")
        eps =0.05 * 255 * 255       # ε最好不要超过0.1，ε较小时，细节层将由一些小的细节组成，例如背景噪音和微小的纹理结构。较大的ε将忽略某些纹理，并特别注意梯度大的边缘。
        radius = 2                  # 引导滤波使用的卷积核半径，滤波窗口变大平滑效果增加，图像更加模糊
        base,mean_a,mean_b = self.guideFilter(imgA, imgA, radius, eps)  # 效果不如官方
        # base = cv2.ximgproc.guidedFilter(guide=imgA, src=imgA, radius=radius, eps=eps)

        # 细节层增强
        detail= imgA - base.astype("float32")
        # detail = abs(detail)
        Gmax = 3
        Gmin = 1 
        im_detail = ((Gmax-Gmin)*(1-mean_a)+Gmin)*detail
        im_detail =np.uint8(cv2.normalize(im_detail,None,0,255,cv2.NORM_MINMAX))

        # 基础层增强
        base =(base).astype("uint16")
        # clahe(限制对比度自适应直方图均衡化)
        # # 直方图整理
        # im_base= Hist16to8(base)
        # 局部直方图
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im_base = clahe.apply(base)
        im_base = np.uint8(cv2.normalize(im_base,None,0,255,cv2.NORM_MINMAX))

        # 图像融合，将基础层分量和细节层分量进行不同比例融合
        alph = 0.4
        fuse = ((1-alph)*im_base+alph*(im_detail)).astype("int16")
        # 归一化
        imgA = np.uint8(cv2.normalize(fuse,None,0,255,cv2.NORM_MINMAX))

        # cv2.imshow("im_base",im_base)
        # cv2.imshow("im_detail",im_detail)
        
        return imgA

    # 灰度提取分段拉伸
    def grey_adjust(self,in_img,low,high,len = 255,bigin = 0):
        dict = (low<=in_img)&(in_img<= high)
        out_img=in_img* (dict)#+high* (in_img > high)+low*(in_img < low)
        out_img=(out_img-low*dict)/(high-low)*len +bigin*dict
        out_img=np.uint8(out_img)
        return out_img

    # 差帧分段线性变换
    def dif_transforms(self,imgA, imgB): 
        # 差帧法
        img = imgB - imgA.astype("int32")
        img =(img).astype("int32")
        # 差值放大
        A = -img*(img<0)
        B = img*(img>0)
        # 分段线性变化，小于15极弱气体，15-100弱小气体，大于100较大气体 
        A = self.grey_adjust(A,low = 3,high = 15,len =130,bigin =0).astype("int32") + self.grey_adjust(A,16,100,125,130) + 255*(A > 101)
        B = self.grey_adjust(B,3,15,130,0).astype("int32") + self.grey_adjust(B,16,100,125,130) + 255*(B > 101)
        # 输出差帧图
        dif_img = np.uint8(cv2.normalize(B-A.astype("int32"),None,0,255,cv2.NORM_MINMAX))
        AB_img = np.uint8(cv2.normalize(B+A.astype("int32"),None,0,255,cv2.NORM_MINMAX))
    
        # 交错差值图
        dif_img = cv2.medianBlur(dif_img,3) 
        return dif_img

    #基础差帧分段线性变换方法
    def DIF_DDE(self,imgA,imgB):
        # 1.原图增强，GF_DDE细节增强
        YY8_imageA = self.GF_DDE(imgA)
        # 2.差帧法增强
        dif_img = self.dif_transforms(imgA,imgB)
        # 3.差帧特征与原图融合
        alpha = 1
        dif_mulLinear_trans = np.uint8(cv2.normalize(alpha*dif_img.astype("float32")+YY8_imageA,None,0,255,cv2.NORM_MINMAX))
        return dif_mulLinear_trans

    #------------------------------#
    #   二、电力常用
    #------------------------------# 
    # 获取图像概率密度
    def get_pdf(self,in_img,value_min,value_max):
        total = in_img.shape[0] * in_img.shape[1]
        Pr=[]
        for i in range(value_min,value_max):
            Pi=np.sum(in_img == i)/total
            Pr.append(Pi)
        return Pr

    # 获取根据直方图计算图像灰度的自适应上下限
    def newcdf_graythresh(self,in_img,low_th=0.01,high_th=0.001,value_min =None,value_max =None):
        total = in_img.shape[0] * in_img.shape[1]
        # 1.获取图像灰度范围，图像减去最小值，减少统计范围
        img_min = int(np.min(in_img))
        img_max = int(np.max(in_img))
        in_img = in_img - img_min
        # 2.计算每个像素值的频率
        T=(in_img.reshape(total))
        hist16 = np.bincount(T)
        # 3.根据输入调整统计范围
        if value_max == None:
            value_max= img_max- img_min
        else:
            value_max = np.max([img_min,value_max]) - img_min
        if value_min == None:
            value_min= img_min- img_min
        else:
            value_min = np.max([img_min,value_min]) - img_min
        hist16[0:value_min] = 0
        hist16[value_max:-1] = 0
        total_num = np.sum(hist16)

        #3.计算上下限的累积像素频率、概率
        L_cumulative_hist16 = np.cumsum(hist16)  
        H_cumulative_hist16 = np.cumsum(hist16[::-1])  
        L_Pr = L_cumulative_hist16/total_num
        H_Pr = H_cumulative_hist16/total_num

        #4.根据累积概率的上下限low_th,high_th找到图像自适应的最大最小值
        min_pix = np.argmin(np.abs(L_Pr - low_th))  + img_min
        max_pix= (H_Pr.shape[0]-np.argmin(np.abs(H_Pr - high_th)))  + img_min
        # print("最大灰度= %s ,最小灰度= %s,灰度宽度= %s" %(max_pix,min_pix,max_pix-min_pix)) 
        return min_pix,max_pix
   
    # 线性拉伸
    def imadjust(self,in_img,low,high):
        out_img=in_img* ((low<=in_img)&(in_img<= high))+high* (in_img > high)+low*(in_img < low)
        out_img=(out_img-low)/(high-low)*255
        out_img=np.uint8(out_img)
        return out_img
    


    def interpolate_color_map(self,data, target_length=255):
        original_length = data.shape[0]

        # 创建目标位置的归一化坐标 [0, ..., 1]
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, target_length)

        # 初始化结果数组
        result = np.zeros((target_length, 3), dtype=np.int32)

        for i in range(target_length):
            # 找到 x_new[i] 在 x_old 中的位置
            idx = np.searchsorted(x_old, x_new[i])

            # 边界情况处理
            if idx == 0:
                result[i] = data[0]
            elif idx == original_length:
                result[i] = data[-1]
            else:
                # 获取左右两个点
                x_left = x_old[idx - 1]
                x_right = x_old[idx]
                frac = (x_new[i] - x_left) / (x_right - x_left)

                # 插值计算
                result[i] = np.round((1 - frac) * data[idx - 1] + frac * data[idx]).astype(np.int32)

        # 确保颜色在 [0, 255] 范围内
        result = np.clip(result, 0, 255)

        return result

    #根据颜色表将灰度转RGB显示
    def ima_to_rgb(self,in_img,RGBlist):

        # RGBlist = self.interpolate_color_map(RGBlist)

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

    #---------------------------------------------------#
    #   局部hdr增强
    #---------------------------------------------------#
    def crop_imhance(self,img_mask,u8_img,u16img,Temperature_list,detail,high_th=0.0065): 
        # 获取根据直方图计算图像灰度的自适应上下限
        crop_u16img = (img_mask[:,:,0]*u16img).astype(np.uint16)
        # 导向滤波划分细节
        detail= (img_mask[:,:,0]*detail).astype(np.float32)
        crop_u16img = (crop_u16img.astype("int32") + 0.8*detail)
        crop_u16img = (crop_u16img - np.min(crop_u16img)).astype("uint16")
        # 获取上下限将16位线性拉伸到8位
        value_min =100
        value_max =None
        if len(Temperature_list) == 11:
            value_max = Temperature_list[ -1,1] +int(np.max(detail)) #80度
            value_min = Temperature_list[ 0,1]  +int(np.min(detail))#-20度  
        # min_pix ,max_pix = self.newcdf_graythresh(crop_u16img,low_th=0.04,high_th=0.0065,value_min =value_min ,value_max =value_max) #局部图中冷反射的占比变大了，低阈值必须调大点
        min_pix ,max_pix = self.newcdf_graythresh(crop_u16img,low_th=0.04,high_th=high_th,value_min =value_min ,value_max =value_max) #局部图中冷反射的占比变大了，低阈值必须调大点
        # min_pix = 7720
        # 温差过小的，不再进行增强
        temp_dif =abs(max_pix-min_pix)
        if temp_dif <100:
            cropimage = (img_mask*u8_img).astype(np.uint8)
        else:
            cropimage = self.imadjust(crop_u16img, min_pix-30,max_pix+50)
            cropimage = self.ima_to_rgb(cropimage,ppbyIron)#灰度转rgb
            cropimage = cv2.cvtColor(cropimage,cv2.COLOR_RGB2BGR)
        # cv2.imshow("cropimage",cropimage)
        return cropimage

    # 根据分割结果裁切图像，局部hdr
    def proMaskCrop(self,original_img,contoursPolygon,img_info = None,high_th=0.0065): 
        # 一、目标掩膜
        img_mask = np.zeros_like(original_img)
        contoursPolygon = np.array(contoursPolygon,dtype=np.int32)
        contoursPolygon = cv2.convexHull(contoursPolygon, clockwise=True)[:,0,:]  # 点集按顺时针排序
        img_mask = cv2.fillPoly(img_mask, [contoursPolygon], color=(1, 1, 1))
        # 二、目标局部增强
        if img_info is  None:
            cropimage = (img_mask*original_img).astype(np.uint8)
        else:
            u16img = img_info[0]
            Temperature_list = img_info[1]
            detail = img_info[2]  
            cropimage = self.crop_imhance(img_mask,original_img,u16img,Temperature_list,detail,high_th) 
        # 三、输出结果
        # 3.1 局部hdr图像
        fuse_image = (np.copy(cropimage) + ((1-img_mask)*original_img)).astype("uint8")
        # 3.2 裁切分割目标外的黑边，减小分辨率，用于检测
        ih,iw = original_img.shape[0],original_img.shape[1]
        x1 = max(min(contoursPolygon[:, 0]), 0)
        y1 = max(min(contoursPolygon[:, 1]), 0)
        x2 = min(max(contoursPolygon[:, 0]), iw)
        y2 = min(max(contoursPolygon[:, 1]), ih)
        cropimage = cropimage[y1:y2, x1:x2]
        # box=[[int(x1),int(y1)],[int(x2),int(y2)]]
        return cropimage,fuse_image

    

    #---------------------------------------------------#
    #   读取图片
    #---------------------------------------------------#
    def keii_read_jpg(self,u16img,Temperature_list):
        #---------------------------------------------------#
        #  细节层
        #---------------------------------------------------#
        # 导向滤波划分细节
        guide_img =(np.copy(u16img)).astype("float32")
        eps =0.2 * 255 * 255       # ε最好不要超过0.1，ε较小时，细节层将由一些小的细节组成，例如背景噪音和微小的纹理结构。较大的ε将忽略某些纹理，并特别注意梯度大的边缘。
        radius = 2                  # 引导滤波使用的卷积核半径，滤波窗口变大平滑效果增加，图像更加模糊
        base0 = cv2.ximgproc.guidedFilter(guide=guide_img, src=guide_img, radius=radius, eps=eps)
        # base0,mean_a,mean_b = guideFilter(guide_img, guide_img, radius, eps)  # 效果不如官方
        # 细节层
        detail= u16img - base0.astype("float32")
        #---------------------------------------------------#
        #  融合层
        #---------------------------------------------------#
        # 细节增强
        fuse_u16 = (u16img + 0.4*detail)
        fuse_u16 =  (fuse_u16 - np.min(fuse_u16)).astype("uint32")
        # 直方图整理，Lowratio越大低灰度拉的越开
        u8img = self.Hist16to8(fuse_u16,Lowratio = 8)
        # 伪彩化
        u8_img=self.ima_to_rgb(u8img,ppbyIron)#灰度转rgb
        img_info = [u16img,Temperature_list,detail]

        return u8_img,img_info

    #---------------------------------------------------#
    #   读取图片
    #---------------------------------------------------#
    def keii_read_dj_jpg(self,dj_temp,Temperature_list):
        u16img = dj_temp*1000
        #---------------------------------------------------#
        #  细节层
        #---------------------------------------------------#
        # 导向滤波划分细节
        guide_img =(np.copy(u16img)).astype("float32")
        eps =0.5 * 255 * 255       # ε最好不要超过0.1，ε较小时，细节层将由一些小的细节组成，例如背景噪音和微小的纹理结构。较大的ε将忽略某些纹理，并特别注意梯度大的边缘。
        radius = 3               # 引导滤波使用的卷积核半径，滤波窗口变大平滑效果增加，图像更加模糊
        base0 = cv2.ximgproc.guidedFilter(guide=guide_img, src=guide_img, radius=radius, eps=eps)
        # 细节层
        detail= u16img - base0.astype("float32")
        #---------------------------------------------------#
        #  基础层
        #---------------------------------------------------#
        # 温度锁定，缩限
        base =np.copy(u16img)
        max = (200)*1000#60度
        min = (-40)*1000  #-20度  
        base[base<min] = min-1
        base[base>max] = max+1
        #---------------------------------------------------#
        #  融合层
        #---------------------------------------------------#
        # 细节增强
        fuse_u16 = (base+ 0.8*detail ).astype("int32") #
        fuse_u16 =  (fuse_u16 - np.min(fuse_u16)).astype("uint32")
        # 获取根据直方图计算图像灰度的自适应上下限
        min_pix ,max_pix = self.newcdf_graythresh(fuse_u16,low_th=0.1,high_th=0,step=0.1*1000,value_min =0 ,value_max =np.max(fuse_u16)) #
        # 获取上下限将16位线性拉伸到8位
        u8img = self.imadjust(fuse_u16, min_pix-30,max_pix+30 )
        # 伪彩化
        u8_img=self.ima_to_rgb(u8img,ppbyIron)#灰度转rgb
        img_info = [u16img,Temperature_list,detail]

        return u8_img,img_info




        



