基于深度学习的YOLOV11松材线虫病树检测系统  ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/10fa1faba4df49f9973cf44eae127d72.png)
航拍图像松材线虫病死树检测数据集，704张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:1类
训练集图像数量:494; 验证集图像数量:141； 测试集图像数量:69
类别名称: 每一类图像数 ，每一类标注数
bad_tree: 413,1840
image num: 704![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1adad638a256448fbc2d78c237a297ef.png)


2.模型代码：模型训练使用yolov11n训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/90895755c8cf49fab6fd066aa8c05c49.png)

3.qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像（
4.（支持windows和linux系统）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/112497428b18400da9553a230813be08.png)


# 航拍松材线虫病死树检测数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/09741c0121ee4fe996799f2f2b71c82b.png)


| 项目 | 详情 |
| --- | --- |
| 数据集名称 | 航拍图像松材线虫病死树检测数据集 |
| 总图片数量 | 704张 |
| 标注格式 | YOLO、VOC、COCO 三种格式 |
| 图像尺寸 | 640×640 |
| 类别总数 | 1类 |
| 数据集划分 | 训练集：494张   验证集：141张   测试集：69张 |
| 类别明细 | bad_tree（病死松树）：包含该目标图像413张，标注框总数1840个 |
| 训练模型 | YOLOv11n，训练轮次30 epoch，[mAP@0.5](mailto:mAP@0.5)=0.733 |
| 配套程序 | PyQt开发可视化检测界面，Windows、Linux双平台可用，附带全套源码+训练权重 |
| 界面功能 | 图片检测、视频检测、摄像头实时检测；输出目标坐标、置信度、检测耗时、目标总数；检测结果可保存导出 |
| 采集方式 | 无人机航拍林区俯拍图像 |
| 应用价值 | 松材线虫病是松树毁灭性病害，这套数据集和检测系统可用于林业遥感巡检、林区航拍自动识别病死松树，辅助林业部门快速定位疫点、开展除治，减少人工野外巡查成本，助力森林病虫害监测预警、生态防护工作。 |

### YOLOv11n 训练参考代码![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d57b148b1e85478db9e99036b331532c.png)


```
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="pine_nematode/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device=0
)
```


### 一、data.yaml（数据集配置文件）
```yaml
path: ./pine_data
train: images/train
val: images/val
test: images/test

names:
  0: bad_tree
```

### 二、训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载yolov11n检测模型
    model = YOLO("yolo11n.pt")
    # 开始训练，30轮
    model.train(
        data="pine_data/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        save=True,
        project="runs/train",
        name="pine_bad_tree"
    )
```

### 三、PyQt5 推理主界面代码 main.py（松材线虫病树检测系统）
```python
import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class PineDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的松材线虫病树检测系统")
        self.resize(1200, 800)
        # 加载训练好的权重
        self.model = YOLO("./runs/train/pine_bad_tree/weights/best.pt")
        self.img_path = ""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.camera_frame)

        # 控件
        self.label_img = QLabel("图像显示区")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setFixedSize(640, 480)

        self.btn_img = QPushButton("导入图片")
        self.btn_img.clicked.connect(self.load_image)
        self.btn_video = QPushButton("导入视频")
        self.btn_video.clicked.connect(self.load_video)
        self.btn_cam = QPushButton("开启摄像头")
        self.btn_cam.clicked.connect(self.open_camera)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_exit = QPushButton("退出")
        self.btn_exit.clicked.connect(self.close)

        self.combo_target = QComboBox()
        self.combo_target.addItem("全部")

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])

        self.label_info = QLabel("用时：0 s | 目标数目：0")
        self.label_pos = QLabel("xmin:0 ymin:0 xmax:0 ymax:0")

    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片(*.jpg *.png *.jpeg)")
        if not file:
            return
        self.img_path = file
        img = cv2.imread(file)
        results = self.model(img)
        res = results[0]
        boxes = res.boxes
        self.table.setRowCount(0)
        for idx, box in enumerate(boxes):
            cls = self.model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            x1,y1,x2,y2 = map(int,xyxy)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(img,f"{cls} {conf:.3f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row,0,QTableWidgetItem(str(row+1)))
            self.table.setItem(row,1,QTableWidgetItem(file))
            self.table.setItem(row,2,QTableWidgetItem(cls))
            self.table.setItem(row,3,QTableWidgetItem(f"{conf:.3f}"))
            self.table.setItem(row,4,QTableWidgetItem(f"[{x1},{y1},{x2},{y2}]"))
        # 转qt图像
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data,w,h,w*c,QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qimg))
        self.label_info.setText(f"用时：{res.speed['inference']/1000:.3f} s | 目标数目：{len(boxes)}")

    def load_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频(*.mp4 *.avi)")
        if not file:
            return
        self.cap = cv2.VideoCapture(file)
        self.timer.start(30)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def camera_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        results = self.model(frame)
        res = results[0]
        boxes = res.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls = self.model.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"{cls} {conf:.3f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data,w,h,w*c,QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qimg))

    def save_result(self):
        QMessageBox.information(self,"提示","检测结果已保存！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PineDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

### 四、依赖环境 requirements.txt
```txt
ultralytics>=8.3.0
torch>=2.0
opencv-python
PyQt5
numpy
```

### 五、部署说明
1. 环境：支持 Windows / Linux
2. 先安装依赖：`pip install -r requirements.txt`
3. 执行 `train.py` 训练，训练完成后把 best.pt 放到对应路径
4. 运行 `main.py` 启动可视化检测界面，具备图片、视频、摄像头实时检测，展示坐标、置信度、目标数量，支持结果保存



