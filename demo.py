from AIDetector_pytorch import Detector
import imutils
import cv2


# 保存模式
# list_txt(path='savelist.txt', list=List1)
# 读取模式
# List_rd = list_txt(path='savelist.txt')
def list_txt(path, list=None):
    '''
    :param path: 储存list的位置
    :param list: list数据
    :return: None/re将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''

    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def main():

    name = 'demo'
    name_list = []
    known_embedding = []
    det = Detector()
    name_list, known_embedding = det.loadIDFeats()

    list_txt(path='name_list.txt', list=name_list)

    fw = open('known_embedding.txt', 'w')
    for line in known_embedding:
        for a in line:
            fw.write(str(a))
            fw.write('\t')
        fw.write('\n')
    fw.close()
    # print(known_embedding)

    cap = cv2.VideoCapture('t1.MOV')
    # cap = cv2.VideoCapture('/Users/hanyingqiao/Desktop/IMG_6757.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None

    while True:


        # RES_DIR = set_res_dir()
        # if TRAIN:
        #     !python train.py --data ../data.yaml --weights yolo5s.pt \
        #     --img 640 --epochs {EPOCHS} --batch-size 16 --name {RES_DIR}

        # try:
        success, im = cap.read()
        import pytest
        # pytest.set_trace()
        if im is None:
            break

        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        import pytest
        if det.isLost is True:
            # pytest.set_trace()
            cv2.imwrite(f'./test-{det.frameCounter/fps}-second.png', result)
            print('lost')

            # todo: quit after write.
        cv2.imshow(name, result)
        cv2.waitKey(t)

        # if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
        #     # 点x退出
        #     break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main()