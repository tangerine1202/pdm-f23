import cv2
import numpy as np
np.set_printoptions(precision=4, suppress=True)

points = [
    # (143, 133),
    # (394, 131),
    # (256, 256),
    # (138, 347),
    # (399, 357),
]


class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        roll, pitch, yaw = phi, theta, gamma

        ### TODO ###
        # Assumption
        # R, t - known pose between top view and front view
        # K - known camera intrinsic matrix and it is the same for both top view and front view
        # n_bev - the normal vector of the plane in bev view
        # d_bev - the distance from the origin to the plane in bev view

        t_bev_W = np.array([[0], [2.5], [0]])
        t_front_W = np.array([[0], [1], [0]])

        t = t_bev_W - t_front_W
        t = -t
        R_front_bev = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(pitch)), -np.sin(np.deg2rad(pitch))],
            [0, np.sin(np.deg2rad(pitch)), np.cos(np.deg2rad(pitch))]
        ])
        R_bev_front = R_front_bev.T

        old_pixels = np.array(points).T
        # homogeneous coordinates of old pixels
        x_bev_h = np.vstack((old_pixels, np.full((1, old_pixels.shape[1]), 1)))
        z_bev = t_bev_W[1]
        z_front = t_front_W[1]
        print('old_pixels_h\n', x_bev_h)

        # 90 fov ~= 2.5m * 2
        # 256px / 1m
        alpha = self.width / 2
        beta = self.height / 2

        # FIXME: K
        K = np.array([
            [alpha, 0, self.width / 2],
            [0, beta, self.height / 2],
            [0, 0, 1]
        ])
        K_inv = np.linalg.inv(K)

        X_bev_C = z_bev * K_inv @ x_bev_h
        print('X_bev_C\n', X_bev_C)
        X_front_C = R_bev_front @ X_bev_C + t
        print('X_front_C from R,t\n', X_front_C)

        # FIXME: hard code, watch out the direction of normal vector
        n_bev_C = np.array([[0], [0], [1]])
        # FIXME: hard code
        d_bev_C = np.abs(n_bev_C.T @ X_bev_C[:, 0])
        # print('n_bev_C.T @ X_bev_C[:, 0]\n', d_bev_C)
        # print('t\n', t)
        # print('t * n_bev_C.T / d_bev_C @ X_bev_C\n', t * n_bev_C.T / d_bev_C @ X_bev_C)

        # X_front_C = (R_bev_front + t * n_bev_C.T / d_bev_C) @ X_bev_C
        # print('X_front_C from H w/o K\n', X_front_C)

        # X_front_C = z_bev * (R_bev_front + t * n_bev_C.T / d_bev_C) @ K_inv @ x_bev_h
        # print('X_front_C from H with K_inv\n', X_front_C)

        H = K @ (R_bev_front + t * n_bev_C.T / d_bev_C) @ K_inv
        # scale factor, z_front * z_bev be will the best
        s = z_bev
        print('scale factor', s)

        x_front_h = H @ x_bev_h
        print('x_front_h w/o s\n', x_front_h)
        x_front_h = s * H @ x_bev_h
        print('x_front_h w/o scale\n', x_front_h)
        x_front_h = x_front_h / x_front_h[2, :]
        print('x_front_h\n', x_front_h)

        # Move the origin from the center of image to the top-left corner
        # x_front_h[0, :] += self.width / 2
        # x_front_h[1, :] += self.height / 2
        # print('x_front_h\n', x_front_h)

        new_pixels = x_front_h[:2, :]
        return new_pixels.T.astype(int)

        return points

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90 + 180

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
