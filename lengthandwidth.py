import cv2
import pyzjr as pz
import pyzjr.Z as Z
from pyzjr.zmath import *
from skimage import measure
import os
from skimage.filters import threshold_otsu,median
from skimage.morphology import skeletonize,dilation,disk
from skimage import io, morphology

class Crack_type():
    """推断裂缝类型"""
    def __init__(self,threshold=3,HWratio=10,Histratio=0.5):
        self.threshold=threshold
        self.HWratio=HWratio
        self.Histratio=Histratio
        self.types = {0: 'Horizontal',
                      1: 'Vertical',
                      2: 'Oblique',
                      3: 'Mesh'}

    def inference_minAreaRect(self,minAreaRect):
        w, h = minAreaRect[1]
        if w > h:
            angle = int(minAreaRect[2])
        else:
            angle = -(90 - int(minAreaRect[2]))
        return w, h, angle

    def classify(self, minAreaRect, skeleton_pts, HW):
        H, W = HW
        w, h, angle = self.inference_minAreaRect(minAreaRect)
        if w / h < self.HWratio or h / w < self.HWratio:
            pts_y, pts_x = skeleton_pts[:, 0], skeleton_pts[:, 1]
            hist_x = np.histogram(pts_x, W)
            hist_y = np.histogram(pts_y, H)
            if self.hist_judge(hist_x[0]) and self.hist_judge(hist_y[0]):
                return 3

        return self.angle2cls(angle)

    def hist_judge(self, hist_v):
        less_than_T = np.count_nonzero((hist_v > 0) & (hist_v <= self.threshold))
        more_than_T = np.count_nonzero(hist_v > self.threshold)
        return more_than_T / (less_than_T + 1e-5) > self.Histratio

    @staticmethod
    def angle2cls(angle):
        angle = abs(angle)
        assert 0 <= angle <= 90, "ERROR: The angle value exceeds the limit and should be between 0 and 90 degrees!"
        if angle < 35:
            return 0
        elif 35 <= angle <= 55:
            return 2
        elif angle > 55:
            return 1
        else:
            return None

class Crack_parameter():
    """计算裂缝参数"""
    def Mesharea(self,img):
        thresh=pz.BinaryImg(img)
        image_array = np.array(thresh)
        white_pixels = np.count_nonzero(image_array)
        return white_pixels

    def detect_crack_areas(self,img, show=True, merge_threshold=3, area_threshold=50):
        """检测裂缝区域并计算裂缝面积。"""
        binary_image = pz.BinaryImg(img)
        connected_image = morphology.closing(binary_image, morphology.disk(merge_threshold))
        labeled_image = measure.label(connected_image, connectivity=2)
        region_props = measure.regionprops(labeled_image)
        area = {}
        crack_label = ord('A')
        Bboxing=[]
        for region in region_props:
            area_value = region.area
            if area_value >= area_threshold:
                minr, minc, maxr, maxc = region.bbox
                Bboxing.append([(minc, minr),(maxc, maxr)])
                if show:
                    pz.Boxcenter_text(img, [minc, minr, maxc, maxr], Z.green, chr(crack_label), Z.red, 0.7, 2)
                if crack_label <= ord('Z'):
                    area[chr(crack_label)] = area_value
                    crack_label += 1
        if show:
            cv2.imshow("Image with Bounding Boxes", img)
            cv2.waitKey(0)
        CrackNum = len(area)
        return area, CrackNum, Bboxing

    def Crop_cracks_img(self,mask, label, Bboxing,show=True):
        cropped_cracks = []
        for bbox in Bboxing:
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            cropped_crack = mask[y1:y2, x1:x2]
            cropped_cracks.append(cropped_crack)
        if show:
            for i, cropped_crack in enumerate(cropped_cracks):
                cv2.imshow(label[i], cropped_crack)
                i += 1
            cv2.waitKey(0)
        return cropped_cracks

    def Piex_coord(self,img, val=255):
        """用于计算所有目标像素的坐标位置,默认为白色"""
        thresh_img = pz.BinaryImg(img)
        y_coords, x_coords = np.where(thresh_img == val)
        coords_list = list(zip(x_coords, y_coords))
        return coords_list

    def Crack_label(self,crackdict):
        label=list(crackdict.keys())
        return label

    def Classify_points_basedon_Bbox(self,gujia_pos, Bboxing):
        """按照边界框对像素进行分类"""
        classified_gujia_pos = {}
        for point in gujia_pos:
            x, y = point
            for i, bbox in enumerate(Bboxing):
                (minc, minr), (maxc, maxr) = bbox
                if minc <= x <= maxc and minr <= y <= maxr:
                    category_label = chr(ord('A') + i)
                    if category_label not in classified_gujia_pos:
                        classified_gujia_pos[category_label] = []
                    classified_gujia_pos[category_label].append(point)
                    break
        return classified_gujia_pos

    def Crack_of_length(self, coords_lst, bias = 1.118):
        """这里用于计算裂缝的长度"""
        connection, inflexion, spaceBetween = 0, 0, 0
        for i in range(len(coords_lst) - 1):
            start = coords_lst[i]
            end = coords_lst[i + 1]
            gap, centre = EuclideanDis(start, end)  # 调用修改后的 EuclideanDis 函数
            if gap == 1.0:
                connection += 1
            elif gap == np.sqrt(2):
                inflexion += 1
            spaceBetween += gap
        return retain(spaceBetween + bias, 2), inflexion, connection

    def Crack_of_width(self, total_area, total_length):
        """求解裂缝平均宽度"""
        ave_width = total_area / (total_length + retain('1e-5'))
        return retain(ave_width,2)

class Crack_skeleton():
    def sketion(self,mode='multifile', input_folder='num', output_folder='output', single_pic='num/001.png'):
        """
        single检测单张图片并保存，multifile检测多张图片并保存
        :param mode: 检测模式  single、multifile
        :param input_folder: 目标文件夹
        :param output_folder: 输出文件夹
        :param single_pic: 用于检测单张图片的路径
        :return: 返回输出文件夹的路径的骨架图
        """
        if mode == 'single':
            image = io.imread(single_pic, as_gray=True)
            # 使用Otsu阈值方法进行二值化处理
            thresh = threshold_otsu(image)
            binary = image > thresh
            skeleton = skeletonize(binary)
            io.imshow(skeleton)
            io.imsave('output.png', skeleton)
            io.show()

        elif mode == 'multifile':
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)  # 如果输出文件夹不存在，就创建它

            for filename in os.listdir(input_folder):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image = io.imread(os.path.join(input_folder, filename), as_gray=True)

                    thresh = threshold_otsu(image)
                    binary = image > thresh
                    binary = dilation(binary, disk(3))
                    binary = median(binary, selem=morphology.disk(5))
                    # 效果不错
                    binary = dilation(binary, disk(2))
                    binary = median(binary, selem=morphology.disk(5))
                    # 添加闭运算
                    selem = morphology.disk(3)
                    binary = morphology.closing(binary, selem)

                    skeleton = skeletonize(binary)

                    output_filename = os.path.join(output_folder, filename)
                    io.imsave(output_filename, skeleton)

            return output_folder

def get_minAreaRect_information(mask):
    mask = pz.BinaryImg(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_merge = np.vstack(contours)
    minAreaRect = cv2.minAreaRect(contour_merge)
    return minAreaRect

def infertype(mask):
    crack=Crack_type()
    H, W = mask.shape[:2]
    mask_copy = mask.copy()
    skeimage, skepoints = pz.ske_information(mask_copy)
    minAreaRect = get_minAreaRect_information(mask)
    result = crack.classify(minAreaRect, skepoints, HW=(H, W))
    crack_type = crack.types[result]
    return crack_type

def Nan():
    return 'Nan'

def reorder(coord,type):
    if type == "Horizontal":
        coords = sorted(coord, key=lambda x: x[0])
        return coords
    elif type == "Vertical" or type == "Oblique":
        return coord


if __name__=="__main__":
    # SECTION 1:加载图片、一定要对应加载裂缝原图与骨架图
    path = r"E:\pythonconda2\dimension2_data\num\046.png"
    skpath = r"E:\pythonconda2\dimension2_data\output\046.png"
    mask = cv2.imread(path)
    mask2 = mask.copy()
    skeimg = cv2.imread(skpath)
    crack = Crack_parameter()

    # SECTION 2:获取裂缝整体的类型
    crack_type = infertype(mask)
    print(f"裂缝总体类型为:{crack_type}")

    # SECTION 3:根据裂缝的不同类型来求解
    if crack_type == 'Mesh':
        white_pixels = crack.Mesharea(mask)
        length = width = num = Nan()
        print(f"裂缝的面积为{white_pixels}，裂缝的长度、宽度、个数均为{length}")
    else:
        # SECTION 4:对裂缝进行个数分类label、故先进行求解裂缝个数、面积、边界框
        area, CrackNum, Bboxing = crack.detect_crack_areas(mask, show=True)
        total_area = sum(area.values())

        print(f"裂缝个数为{CrackNum},总面积为{total_area},裂缝的面积为:{area},每个边界框的值{Bboxing}")
        label = crack.Crack_label(area)
        print("裂缝的label:", label)
        # SECTION 5:依据边界框，裁剪裂缝区域
        cropped_cracks = crack.Crop_cracks_img(mask2, label, Bboxing, show=False)
        # SECTION 6:获取裁剪图像的类型，再进行下一步判断
        croped_list = []
        for i, crop in enumerate(cropped_cracks):
            crop_type = infertype(crop)
            croped_list.append([label[i], crop_type])
            i += 1
        print(croped_list)
        # SECTION 7:获取骨架图目标像素坐标，并通过Bbox对像素点进行分类
        gujia_pos = crack.Piex_coord(skeimg)
        classified_gujia_pos = crack.Classify_points_basedon_Bbox(gujia_pos, Bboxing)
        # print(classified_gujia_pos)
        # SECTION 8:按照类型对classified_gujia_pos中的骨架进行
        new_classified_gujia_pos = {}
        for item in croped_list:
            labels, crack_type = item
            new_classified_gujia_pos[labels] = reorder(classified_gujia_pos[labels], crack_type)
        # print(new_classified_gujia_pos)
        # SECTION 9:按照字典的key去计算裂缝的长度和宽度
        all_length = []
        for key in label:
            spaceBetween, inflexion, connection = crack.Crack_of_length(new_classified_gujia_pos[key])
            all_length.append(spaceBetween)
            print(f"{key},裂缝长度:{round(spaceBetween, 2)},拐点:{inflexion},连通:{connection}")
        total_length = sum(all_length)
        ave_width = crack.Crack_of_width(total_area, total_length)
        print(f"裂缝总长度{total_length},总面积{total_area},平均宽度{ave_width}")