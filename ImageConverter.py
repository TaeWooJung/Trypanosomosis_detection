import cv2
import os


def video2png(vid):

    # Read the video from the specified path

    try:
        # Create an appropriately named folder
        if not os.path.exists(vid[:len(vid)-4] + '_frames'):
            os.makedirs(vid[:len(vid)-4] + '_frames')

    # If failure to create the file should occur, send out an error message
    except OSError:
        print('Error: was not able to create file')

    # Read the video
    cam = cv2.VideoCapture(vid)
    # Frame counter
    currentframe = 1

    while True:
        # Reading from frame
        ret, frame = cam.read()

        if ret:
            leading_zero = '0'*(3-len(str(currentframe)))
            # If there are video frames left continue creating images

            if currentframe % 2 != 0:

                name = './' + vid[:len(vid)-4] + '_frames/' + 'vid{}'.format(vid[len(vid)-6:len(vid)-4]) + '_' + leading_zero + str(currentframe) + '.png'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1

        else:
            break

    # Release all space and windows once completed
    cam.release()
    cv2.destroyAllWindows(),


# Negative sample video conversion
# 사용법: ImageConverter.py가 위치한 directory에서 data라는 directory에 Negative sample video들을 넣고 이름을 n_vidxx
# 로 이름을 바꾼 후에 이 코드를 돌리면 됩니다.
videos = [name for name in os.listdir('data') if 'vid' in name]
print(videos)

for vid in videos:

    DIR = 'data/' + vid
    video2png(DIR)
