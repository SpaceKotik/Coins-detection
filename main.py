import numpy as np
import cv2 as cv
import glob
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



def run_main():

    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap = cv.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    fourcc = cv.VideoWriter_fourcc(*'H264')
    cap.set(cv.CAP_PROP_FOURCC, fourcc)
    cap.set(cv.CAP_PROP_SATURATION, 100)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 140)


    #clf = load_model()
    clf = train_model()

    while True:
        ret, frame = cap.read()
        #roi = frame[0:500, 0:500]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


        clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray_end = clahe.apply(gray)
        gray_end = cv.GaussianBlur(gray_end, (9, 9), 0)



        gray_blur = cv.GaussianBlur(gray, (11, 11), 0)

        thresh = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY_INV, 11, 1)
        kernel = np.ones((3, 3), np.uint8)
        #closing = cv.morphologyEx(thresh, cv.MORPH_ERODE,
        #                           kernel, iterations=1)
        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE,
                                  kernel, iterations=1)

        contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL,
                                               cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 1500 or area > 9000:
                continue
            if len(cnt) < 5:
                continue
            ellipse = cv.fitEllipse(cnt)
            axes = ellipse[1]
            minor, major = axes
            if (major/minor < 1.5):
                cv.ellipse(gray_blur, ellipse, (0, 255, 255), 2)
        cv.imshow('new', closing)
        cv.imshow('blur', gray_blur)







        rows = gray.shape[0]
        circles = cv.HoughCircles(gray_end, cv.HOUGH_GRADIENT, 1, rows / 11, param1=250, param2=35, minRadius=15,
                                    maxRadius=int(rows / 5))

        count = 0
        total = 0
        if circles is not None:
            diameter = []
            materials = []
            coordinates = []

            # append radius to list of diameters (we don't bother to multiply by 2)
            for (x, y, r) in circles[0, :]:
                diameter.append(r)

            # convert coordinates and radii to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over coordinates and radii of the circles
            for (x, y, d) in circles:
                count += 1

                # add coordinates to list
                coordinates.append((x, y))

                # extract region of interest
                roi = frame[y - d:y + d, x - d:x + d]

                # try recognition of material type and add result to list
                material = predictMaterial(roi, clf)
                materials.append(material)

                # draw contour and results in the output image
                cv.circle(frame, (x, y), d, (0, 255, 0), 2)
                cv.putText(frame, material,
                           (x - 40, y), cv.FONT_HERSHEY_PLAIN,
                           1.5, (0, 255, 0), thickness=2, lineType=cv.LINE_AA)

            # get biggest diameter
            biggest = max(diameter)
            i = diameter.index(biggest)

            # scale everything according to maximum diameter
            # todo: this should be chosen by the user
            if materials[i] == "Label":
                diameter = [x / biggest * 42.0 for x in diameter]
                scaledTo = "Scaled to Label"
            if materials[i] == "Gold":
                diameter = [x / biggest * 22.0 for x in diameter]
                scaledTo = "Scaled to Gold"
            elif materials[i] == "Silver":
                diameter = [x / biggest * 25.0 for x in diameter]
                scaledTo = "Scaled to Silver"
            elif materials[i] == "Silver":
                diameter = [x / biggest * 23.0 for x in diameter]
                scaledTo = "Scaled to Silver"
            elif materials[i] == "Silver":
                diameter = [x / biggest * 20.5 for x in diameter]
                scaledTo = "Scaled to Silver"
            elif materials[i] == "Bronze":
                diameter = [x / biggest * 19.5 for x in diameter]
                scaledTo = "Scaled to Bronze"
            else:
                scaledTo = "Unable to scale"

            i = 0
            total = 0
            while i < len(diameter):
                d = diameter[i]
                m = materials[i]
                (x, y) = coordinates[i]
                t = "Unknown"

                # compare to known diameters with some margin for error
                if math.isclose(d, 25.0, abs_tol=1.25) and m == "Silver":
                    t = "5 rub"
                    total += 5
                elif math.isclose(d, 23.0, abs_tol=1.75) and m == "Silver":
                    t = "2 rub"
                    total += 2
                elif math.isclose(d, 19.5, abs_tol=1.75) and m == "Silver":
                    t = "1 rub"
                    total += 1
                elif math.isclose(d, 17.5, abs_tol=2.5) and m == "Silver":
                    t = "0.05 rub"
                    total += 0.05
                elif math.isclose(d, 22.0, abs_tol=2.75) and m == "Gold":
                    t = "10 rub"
                    total += 10
                elif math.isclose(d, 18.5, abs_tol=1.75) and m == "Gold":
                    t = "0.50 rub"
                    total += 0.50
                elif math.isclose(d, 16.5, abs_tol=2.75) and m == "Gold":
                    t = "0.10 rub"
                    total += 0.10

                # write result on output image
                cv.putText(frame, t,
                           (x - 40, y + 22), cv.FONT_HERSHEY_PLAIN,
                           1.8, (0, 0, 255), thickness=2, lineType=cv.LINE_AA)
                i += 1

        cv.putText(frame, "Coins detected: {}, RUB {:2}".format(count, total),
                   (5, frame.shape[0] - 24), cv.FONT_HERSHEY_PLAIN,
                   1.0, (0, 0, 255), lineType=cv.LINE_AA)

        cv.imshow("Gray", gray_end)
        cv.imshow('Contours', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()



def run_samples():

    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap = cv.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    #fourcc = cv.VideoWriter_fourcc(*'H264')
    #cap.set(cv.CAP_PROP_FOURCC, fourcc)
    cap.set(cv.CAP_PROP_SATURATION, 100)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 140)

    clf = load_model()

    currFile = 0

    while True:


        ret, frame = cap.read()
        output = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray = cv.medianBlur(gray, 11)

        # make picture more contrast
        cv.imshow("gray", gray)

        rows = gray.shape[0]
        clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv.GaussianBlur(gray, (9, 9), 0)

        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 11, param1=250, param2=35, minRadius=15,
                                  maxRadius=int(rows / 3))

        if circles is not None:
            currFile += 1

            detected_circles = np.uint16(np.around(circles))
            for (x, y, r) in detected_circles[0, :]:
                cv.circle(output, (x, y), r, (255, 0, 0), 2)

            # todo: refactor
            diameter = []
            materials = []
            coordinates = []

            count = 0
            # append radius to list of diameters (we don't bother to multiply by 2)
            for (x, y, r) in circles[0, :]:
                diameter.append(r)

            # convert coordinates and radii to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over coordinates and radii of the circles
            for (x, y, d) in circles:
                count += 1

                # add coordinates to list
                coordinates.append((x, y))

                # extract region of interest
                roi = frame[y - d:y + d, x - d:x + d]

                # write masked coin to file
                if True:
                    m = np.zeros(roi.shape[:2], dtype="uint8")
                    w = int(roi.shape[1] / 2)
                    h = int(roi.shape[0] / 2)
                    cv.circle(m, (w, h), d, (255), -1)
                    maskedCoin = cv.bitwise_and(roi, roi, mask=m)
                    if flag:
                        cv.imwrite("extracted/{}coin{}.png".format(currFile, count), maskedCoin)


        cv.imshow('output', output)
        flag = False
        if cv.waitKey(1) & 0xFF == ord('s'):
            print('flag!')
            flag = True
        elif cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def calcHistogram(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv.circle(m, (w, h), 60, 255, -1)
    #m = cv.GaussianBlur(m, (3, 3), 0)

    # calcHist expects a list of images, color channels, mask, bins, ranges
    #h = cv.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    h = cv.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 384, 0, 384, 0, 384])
    # return normalized "flattened" histogram
    return cv.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv.imread(file)
    return calcHistogram(img)


# define Enum class
class Enum(tuple): __getattr__ = tuple.index

# Enumerate material types for use in classifier
Material = Enum(('Bronze', 'Silver', 'Gold', 'Label'))

def load_model():
    clf = joblib.load('trained_model.pkl')
    return clf


def train_model():
    sample_images_Gold = glob.glob("samples/real/Gold/*")
    sample_images_Silver = glob.glob("samples/real/Silver/*")
    sample_images_Bronze = glob.glob("samples/real/Bronze/*")

    x = []
    y = []

    for i in sample_images_Bronze:
        x.append(calcHistFromFile(i))
        y.append(Material.Bronze)

    for i in sample_images_Gold:
        x.append(calcHistFromFile(i))
        y.append(Material.Gold)

    for i in sample_images_Silver:
        x.append(calcHistFromFile(i))
        y.append(Material.Silver)

    # clf = MLPClassifier(solver="lbfgs")
    clf = MLPClassifier(solver="adam", hidden_layer_sizes=(100), activation='relu', max_iter=600, verbose=True)

    # split samples into training and test data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # train and score classifier
    clf.fit(X_train, y_train)
    score = int(clf.score(X_test, y_test) * 100)
    print("Classifier mean accuracy: ", score, "%")

    joblib.dump(clf, 'trained_model.pkl')
    return clf


def predictMaterial(roi, clf):
    # calculate feature vector for region of interest
    hist = calcHistogram(roi)
    # predict material type
    s = clf.predict([hist])
    # return predicted material type
    return Material[int(s)]


if __name__ == "__main__":
    run_main()
    #run_samples()

