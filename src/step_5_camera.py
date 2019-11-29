import cv2
import numpy as np
import onnxruntime as ort


def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]


def main():
    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')

    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = center_crop(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = (frame - 0.485) / 0.229

        x = cv2.resize(x, (28, 28))
        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("you", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
