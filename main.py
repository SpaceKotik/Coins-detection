import numpy as np
import cv2 as cv
import glob
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# define Enum class
class Enum(tuple):
    __getattr__ = tuple.index


# Enumerate material types for use in classifier
Material = Enum(('Bronze', 'Silver', 'Gold'))


def run_main():
    clf = load_model()
    #clf = train_model()

    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap = cv.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv.CAP_PROP_SATURATION, 100)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 160)
    cap.set(cv.CAP_PROP_FPS, 1)

    while True:
        ret, frame = cap.read()
        # roi = frame[0:500, 0:500]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray_end = clahe.apply(gray)
        gray_end = cv.GaussianBlur(gray_end, (9, 9), 0)

        gray_blur = cv.GaussianBlur(gray, (11, 11), 0)

        thresh = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 1)
        kernel = np.ones((3, 3), np.uint8)
        # closing = cv.morphologyEx(thresh, cv.MORPH_ERODE,
        #                           kernel, iterations=1)
        # closing = cv.morphologyEx(closing, cv.MORPH_DILATE,
        #                          kernel, iterations=1)
        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE,
                                  kernel, iterations=1)

        contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        circles = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 1500 or area > 7000:
                continue
            if len(cnt) < 5:
                continue
            ellipse = cv.fitEllipse(cnt)
            axes = ellipse[1]
            minor, major = axes
            if major/minor < 1.5:
                circles.append(ellipse)
                cv.ellipse(gray_blur, ellipse, (0, 255, 255), 2)
        cv.imshow('Adaptive threshold', closing)
        cv.imshow('Blur', gray_blur)

        ellipses = []
        for circ in circles:
            coords = circ[0]
            x, y = coords
            axes = circ[1]
            minor, major = axes
            ellipses.append([x, y, (major + minor)/4.0])

        count = 0
        total = 0
        if len(ellipses) > 0:
            diameter = []
            materials = []
            coordinates = []

            # append radius to list of diameters (we don't bother to multiply by 2)
            for ell in ellipses:
                diameter.append(ell[2])

            # convert coordinates and radii to integers
            circles = np.round(ellipses).astype("int")

            # loop over coordinates and radii of the circles
            for (x, y, diam) in circles:
                count += 1

                # add coordinates to list
                coordinates.append((x, y))

                # extract region of interest
                roi = frame[y - diam:y + diam, x - diam:x + diam]

                # try recognition of material type and add result to list
                material = predict_material(roi, clf)
                materials.append(material)

                # draw contour and results in the output image
                cv.circle(frame, (x, y), diam, (0, 255, 0), 2)
                cv.putText(frame, material,
                           (x - 40, y), cv.FONT_HERSHEY_PLAIN,
                           1.5, (0, 255, 0), thickness=2, lineType=cv.LINE_AA)

            # get biggest diameter
            biggest = max(diameter)
            i = diameter.index(biggest)

            # scale everything according to maximum diameter
            if materials[i] == "Gold":
                diameter = [x / biggest * 22.0 for x in diameter]
                scaledTo = "Scaled to Gold"
            elif materials[i] == "Silver":
                diameter = [x / biggest * 25.0 for x in diameter]
                scaledTo = "Scaled to Silver"
            elif materials[i] == "Bronze":
                diameter = [x / biggest * 19.5 for x in diameter]
                scaledTo = "Scaled to Bronze"
            else:
                scaledTo = "Unable to scale"

            i = 0
            total = 0
            while i < len(diameter):
                diam = diameter[i]
                mat = materials[i]
                (x, y) = coordinates[i]
                coin_quantity = "Unknown"

                # compare to known diameters with some margin for error
                if math.isclose(diam, 25.0, abs_tol=1.5) and mat == "Silver":
                    coin_quantity = "5 rub"
                    total += 5
                elif math.isclose(diam, 23.0, abs_tol=1.75) and mat == "Silver":
                    coin_quantity = "2 rub"
                    total += 2
                elif math.isclose(diam, 20.5, abs_tol=1.55) and mat == "Silver":
                    coin_quantity = "1 rub"
                    total += 1
                elif math.isclose(diam, 18.5, abs_tol=2.5) and mat == "Silver":
                    coin_quantity = "0.05 rub"
                    total += 0.05
                elif math.isclose(diam, 22.0, abs_tol=1.75) and (mat == "Bronze" or mat == "Gold"):
                    coin_quantity = "10 rub"
                    total += 10
                elif math.isclose(diam, 19.5, abs_tol=1.5) and (mat == "Bronze" or mat == "Gold"):
                    coin_quantity = "0.50 rub"
                    total += 0.50
                elif math.isclose(diam, 17.5, abs_tol=3.75) and (mat == "Bronze" or mat == "Gold"):
                    coin_quantity = "0.10 rub"
                    total += 0.10

                # write result on output image
                cv.putText(frame, coin_quantity,
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
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap = cv.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(cv.CAP_PROP_SATURATION, 100)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 140)

    curr_file = 0
    flag = False
    while True:
        ret, frame = cap.read()
        output = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray = cv.medianBlur(gray, 11)

        # make picture more contrast
        cv.imshow("gray", gray)

        clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv.GaussianBlur(gray, (9, 9), 0)

        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 11, param1=250, param2=35, minRadius=15,
                                  maxRadius=int(rows / 3))

        if circles is not None:
            curr_file += 1

            detected_circles = np.uint16(np.around(circles))
            for (x, y, r) in detected_circles[0, :]:
                cv.circle(output, (x, y), r, (255, 0, 0), 2)

            coordinates = []

            count = 0

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
                        cv.imwrite("extracted/{}coin{}.png".format(curr_file, count), maskedCoin)


        cv.imshow('output', output)
        flag = False
        if cv.waitKey(1) & 0xFF == ord('s'):
            flag = True
        elif cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def calc_histogram(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv.circle(m, (w, h), 60, 255, -1)
    # m = cv.GaussianBlur(m, (3, 3), 0)

    # calcHist expects a list of images, color channels, mask, bins, ranges
    h = cv.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 384, 0, 384, 0, 384])
    # return normalized flattened histogram
    return cv.normalize(h, h).flatten()


def calc_hist_from_file(file):
    img = cv.imread(file)
    return calc_histogram(img)


def load_model():
    clf = joblib.load('trained_model.pkl')
    return clf


def train_model():
    sample_images_gold = glob.glob("samples/real/Gold/*")
    sample_images_silver = glob.glob("samples/real/Silver/*")
    sample_images_bronze = glob.glob("samples/real/Bronze/*")

    x = []
    y = []

    for i in sample_images_bronze:
        x.append(calc_hist_from_file(i))
        y.append(Material.Bronze)

    for i in sample_images_gold:
        x.append(calc_hist_from_file(i))
        y.append(Material.Gold)

    for i in sample_images_silver:
        x.append(calc_hist_from_file(i))
        y.append(Material.Silver)

    # clf = MLPClassifier(solver="lbfgs")
    clf = MLPClassifier(solver="adam", hidden_layer_sizes=100, activation='relu', max_iter=600, verbose=True)

    # split samples into training and test data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # train and score classifier
    clf.fit(X_train, y_train)
    score = int(clf.score(X_test, y_test) * 100)
    print("Classifier mean accuracy: ", score, "%")

    joblib.dump(clf, 'trained_model.pkl')
    return clf


def predict_material(roi, clf):
    # calculate feature vector for region of interest
    hist = calc_histogram(roi)
    # predict material type
    s = clf.predict([hist])
    # return predicted material type
    return Material[int(s)]


if __name__ == "__main__":
    run_main()
    # run_samples()