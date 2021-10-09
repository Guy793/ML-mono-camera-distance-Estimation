
from DataLoader import *
import cv2



args,ap=Build_parser()
annotations_file_path=args.annotations_file_path
img_dir=args.img_dir
flight_dir=args.flight_dir
# print(args.annotations_file_path)
flight_ids = check_available_flights(flight_dir)
print(flight_ids)
lucky_flight_id = random.choice(flight_ids)
lucky_flight_id = flight_ids[0]
print('chosen flight id', lucky_flight_id)
# #simple exmple about using dataset
dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir, flight_id=lucky_flight_id)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

images, Range,is_above_Horizon,bounding_box=next(iter(data_loader))

def Returned_segment_object(images_batch,bounding_box):
    masked_list=[]
    for i in range(len(images_batch)):
        left = bounding_box[i][0]
        top = bounding_box[i][1]
        right = bounding_box[i][2]
        buttom = bounding_box[i][3]
        mask = np.zeros(images_batch[i].shape[:2], dtype="uint8")
        cv2.rectangle(mask, (left, top), (right + left, buttom + top), 255, -1)
        masked = cv2.bitwise_and(np.uint8(images_batch[i]), np.uint8(images_batch[i]), mask=mask)
        masked_list.append(masked)
    return masked_list

# bounding_box=bounding_box[300]#bounding_box.squeeze()
# left=bounding_box[0]#bounding box shape is [left.top,width,hight]
# top=bounding_box[1]
# right=bounding_box[2]
# buttom=bounding_box[3]
#
#
# #show clean image
# images=images[300]
# plt.imshow(images)
# plt.show()
frame, Range, is_above_Horizon, bounding_box=next(iter(data_loader))
print(frame.squeeze().shape)
mask = np.zeros_like(frame.squeeze())

prev_gray = cv2.cvtColor(np.float32(frame.squeeze()), cv2.COLOR_BGR2GRAY)

for frame, Range, is_above_Horizon, bounding_box in data_loader:
    frame=frame.squeeze()
    # Opens a new window and displays the input
    # frame
    cv2.imshow("input", np.uint8(frame))
    # Converts each frame to grayscale - we previously
    # only converted the first frame to grayscale
    gray = cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 20, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print('optical flow magnitude',magnitude)
    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv2.imshow("dense optical flow", rgb)

    # Updates previous frame
    prev_gray = gray

    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break