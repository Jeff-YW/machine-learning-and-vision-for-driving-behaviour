## Saved Results

For a comprehensive understanding of our results, we have saved several video files. Due to the file size, these videos are hosted on OneDrive. You can view them by following the links below:

### Video Links

- [Project Output Video](https://liveuclac-my.sharepoint.com/:v:/g/personal/zcemwei_ucl_ac_uk/EZZUSsAtEtNGg9ffswYKZ2wBpR6z7NbeQQT1oJeNTdd46A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0RpcmVjdCJ9fQ&e=zTy8Lo)
- [Binary Lane Video](https://liveuclac-my.sharepoint.com/:v:/g/personal/zcemwei_ucl_ac_uk/EbYVjCi_OaFPv0OTQK80DaIBI3V-5CqAXF4OemAG-UyNnw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0RpcmVjdCJ9fQ&e=6V8wUH)
- [Bird-eye View Video](https://liveuclac-my.sharepoint.com/:v:/g/personal/zcemwei_ucl_ac_uk/EbE8XL-RCA9KmMkyCepAdMcBcZSKmDNLjqm4nDcx5SGY6A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0RpcmVjdCJ9fQ&e=h7wxdh)


# Lane Detection Procedure

The lane detection performance is shown by the example output videos in "./examples/project_video_output.mp4"

## Restore Bird-eye view videos
- Using the calibration parameter of the camera found by chessboard pictures, we apply the `warp(img)` function with opencv `cv2.undistort` and `cv2.warpPerspective` to produce 
the bird-eye view frames of the original distorted video.

## Thresholding
- apply sobel line detection method, find the lines (or contours of objects) in the frames. Then thresholding the frames for coloured lanes (i.e. white and yellow).

## Lane Properties Calculation
- This is done in Class `Line()` function `add_fit(*args, **kargs)`. The split images (`limg` and `rimg`) contains the single 
left lane and right lane of the recorded road, respectively. 

### Checking for Nonzero Points:
- The function starts by checking if there are any nonzero points in the images. 
- Nonzero points in a binary image indicate the presence of features of interest, lane markings.

### Polynomial fitting:
- If lanes are detected, it performs a second-order polynomial fit (using `np.polyfit`) for the coordinates of the nonzero points. 
- This fitting is scaled by `mx` and `my`, which are the scaling factors to adjust the coordinate system from pixels to real-world units.

### Calculation of Base Point: 
- The base point of the lane (where the lane meets the bottom of the image) is calculated. 
- This involves scaling the height of the image (`self.limg.shape[0]` or `self.rimg.shape[0]`) by `my`.
- Note that the horizontal axis is `x` (i.e. the 1 index of array shape) and the vertical axis is `y` (i.e. the 0 index of array shape).

### Radius of curvature:
- The radius of curvature of the lane at the base point is calculated using the formula for 
- The radius of curvature of a curve given by a second-order polynomial. This formula is `R = [1 + (2ay + b)^2]^(3/2) / |2a|`, 
where `a` and `b` are the coefficients of the quadratic polynomial, 
and `y` is the vertical coordinate of the point of interest (in this case, the base of the lane).

### Lane Position Calculation:
- `self.lcurrent_fit[0] * ly_base ** 2 + self.lcurrent_fit[1] * ly_base ** 1 + self.lcurrent_fit[2] * ly_base ** 0`

- This part of the equation is a polynomial evaluation. The `self.lcurrent_fit` array contains the coefficients 
of a second-order polynomial (quadratic equation) that models the left lane's shape in the image.

- The coefficients are `[a, b, c]`, corresponding to the quadratic, linear, and constant terms of the polynomial `a*y^2 + b*y + c`.
ly_base is the vertical coordinate at the bottom of the image (the base of the lane), scaled by a factor my to 
convert it from pixels to some real-world unit like meters.

- This polynomial is evaluated at `ly_base` to find the horizontal position of the lane at the bottom of the image.


- The term `mx * float(self.limg.shape[1]) / 2` calculates the horizontal midpoint of the image.

- By subtracting this from the polynomial evaluation, the equation gives the horizontal position of the base point relative to the image's center.


## Best Fit Lane

### Functionality
- The `best_fit` function refines lane line detection over multiple iterations.
- It ensures the reliability of the detected lanes and calculates key parameters like curvature radius and lane position.

### Usage
- Input your pre-processed binary images with initial lane line detections.
- Set appropriate `mx` and `my` values to scale pixel dimensions to real-world units.
- Call the function after each detection iteration for refined results.

### Output
- Refined polynomial coefficients for both left and right lane lines.
- Calculated radius of curvature and lane base positions.
- Overall lane curvature and vehicle offset from the lane center.

### Integration
Integrate this module into your lane detection pipeline for enhanced accuracy and reliability in lane tracking and vehicle positioning.
