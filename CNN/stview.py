import pickle
import math
import numpy
import requests
import io
import time

import PIL
import PIL.Image

import cv2
import skimage
import skimage.exposure
import skimage.transform

import colorsys
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

import theano
import theano.tensor
import theano.tensor as T

import googlemaps
import googlemaps.client
import googlemaps.convert
import googlemaps.directions
import googlemaps.distance_matrix
import googlemaps.elevation
import googlemaps.exceptions
import googlemaps.geocoding
import googlemaps.roads
import googlemaps.timezone

import polyline
import polyline.codec

import webbrowser
import pygmaps.pygmaps

import CNN
import CNN.prop
import CNN.conv
import CNN.utils
import CNN.enums
import CNN.nms


def span_google_street_view():
    api_key = __read_api_key()
    client = googlemaps.client.Client(key=api_key)

    # convert start/stop addresses to geo-locations
    address11 = '6 Longmead Road, Townhill Park, Southampton, UK'
    address12 = 'Portswood Street, Southampton, UK'
    address13 = 'Sainsbury\'s, 224 Portswood Road, Southampton SO17 2LB, United Kingdom'

    address21 = 'University of Southampton, Highfield Campus, Southampton, UK'
    address22 = 'Jurys Inn Southampton, Charlotte Place, Southampton SO14 0TB, United Kingdom'

    geocode_start = googlemaps.client.geocode(client, address22)
    geocode_stop = googlemaps.client.geocode(client, address13)

    start_location = geocode_start[0]["geometry"]["location"]
    stop_location = geocode_stop[0]["geometry"]["location"]

    # get the direction from start to stop, get them in terms of geo-location points // driving
    direction_result = googlemaps.client.directions(client, start_location, stop_location, mode="driving")

    # decode the polyline of the direction to get the points
    points = polyline.codec.PolylineCodec().decode(direction_result[0]["overview_polyline"]["points"])
    locations = __convert_points_to_locations(points)
    locations = __calculate_heading(locations)
    __plot_points_on_map(locations)

    # missing steps
    # generate more points, adjust the pace, then calculate the heading

    __show_street_view_images(locations, api_key)

    x = 10
    return

    # since we interpolated points in the direction, these generated points might not be on
    # the road (if road wasn't straight line). The solution is to snap these point to the road
    # path = [(start_location_lat, start_location_lng), (stop_location_lat, stop_location_lng)]
    # road_locations = googlemaps.client.snap_to_roads(client, path, interpolate=True)

    # load the models once and for all
    prohib_recog_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_p_80.pkl"
    mandat_recog_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_m_80.pkl"
    prohib_detec_model_path = "D:\\_Dataset\\GTSDB\\las_model_p_80_binary.pkl"
    mandat_detec_model_path = "D:\\_Dataset\\GTSDB\\las_model_m_80_binary.pkl"
    superclass_recognition_model_path = "D:\\_Dataset\\SuperClass\\cnn_model_28_lasagne.pkl"

    detect_net_p = __build_detector(prohib_recog_model_path, prohib_detec_model_path, batch_size)
    detect_net_m = __build_detector(mandat_recog_model_path, mandat_detec_model_path, batch_size)
    recog_superclass_cnn = __build_classifier(superclass_recognition_model_path)

    # detect the region, using superclass-specific recognition model
    regions_p = __detect(img_color, batch_size, detect_net_p)
    regions_m = __detect(img_color, batch_size, detect_net_m)

    # extract regions from the image (for each region, extract different scales)
    # then do super-class classification for these regions
    # don't forget to classify and get the prediction probability
    # ppp = net_cnn.predict_proba(imgs)

    img_color = cv2.imread("D:\\_Dataset\\GTSDB\\Test_PNG\\_img15.png")
    batch_size = 100



    # after detection, run super-class classification on only the strong regions
    strong_regions = regions

    __save_detection_result(img_color, regions)


# region Detector

def __build_detector(recognition_model_path, detection_model_path, batch_size):
    # stack the regions of all the scales in one array
    # please note that a scale can have no regions, so using vstack wouldn't work
    # remove the scales with empty regions then use vstack

    ##############################
    # Build the detector         #
    ##############################

    loaded_objects = CNN.utils.load_model(model_path=recognition_model_path, model_type=CNN.enums.ModelType._02_conv3_mlp2)
    img_dim = loaded_objects[1]
    kernel_dim = loaded_objects[2]
    nkerns = loaded_objects[3]
    pool_size = loaded_objects[5]

    layer0_W = theano.shared(loaded_objects[6], borrow=True)
    layer0_b = theano.shared(loaded_objects[7], borrow=True)
    layer1_W = theano.shared(loaded_objects[8], borrow=True)
    layer1_b = theano.shared(loaded_objects[9], borrow=True)
    layer2_W = theano.shared(loaded_objects[10], borrow=True)
    layer2_b = theano.shared(loaded_objects[11], borrow=True)

    layer0_input = T.tensor4(name='input')
    layer0_img_dim = img_dim
    layer0_img_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
    layer0_kernel_dim = kernel_dim[0]
    layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
    layer1_kernel_dim = kernel_dim[1]
    layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
    layer2_kernel_dim = kernel_dim[2]
    layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
    layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

    # layer 0, 1, 2: Conv-Pool
    layer0_output = CNN.conv.convpool_layer(
        input=layer0_input, W=layer0_W, b=layer0_b,
        image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
        filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
        pool_size=pool_size
    )
    layer1_output = CNN.conv.convpool_layer(
        input=layer0_output, W=layer1_W, b=layer1_b,
        image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
        filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
        pool_size=pool_size
    )
    layer2_output = CNN.conv.convpool_layer(
        input=layer1_output, W=layer2_W, b=layer2_b,
        image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
        filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
        pool_size=pool_size
    )
    # do the filtering using 3 layers of Conv+Pool
    conv_fn = theano.function([layer0_input], layer2_output)

    # load the regression model
    with open(detection_model_path, 'rb') as f:
        nn_mlp = pickle.load(f)

    return conv_fn, nn_mlp, layer3_input_shape


def __detect(img_color, batch_size, net):
    """
    detect a traffic sign form the given natural image
    detected signs depend on the given model, for example if it is a prohibitory detection model
    we'll only detect prohibitory traffic signs
    :param img_path:
    :param model_path:
    :param classifier:
    :param img_dim:
    :return:
    """

    net_cnn, net_mlp, net_mlp_input_shape = net

    ##############################
    # Extract detection regions  #
    ##############################

    # pre-process image by: equalize histogram and stretch intensity
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img = img.astype(float) / 255.0
    img_dim = 80

    # min, max defines what is the range the detection proposals
    max_window_dim = int(img_dim * 2)
    min_window_dim = int(img_dim / 4)

    # regions, locations and window_dim at each scale
    regions = []
    locations = []
    window_dims = []
    r_count = 0

    # important, instead of naively add every sliding window, we'll only add
    # windows that covers the strong detection proposals
    prop_weak, prop_strong, prop_map, prop_circles = CNN.prop.detection_proposal(img_color, min_dim=min_window_dim, max_dim=max_window_dim)
    if len(prop_strong) == 0:
        print("... NO TRAFFIC SIGN PROPOSALS WERE FOUND")
        return [], [], [], []

    # loop on the detection proposals
    scales = numpy.arange(0.7, 1.58, 0.05)
    for prop in prop_strong:
        x1 = prop[0]
        y1 = prop[1]
        x2 = prop[2]
        y2 = prop[3]
        w = x2 - x1
        h = y2 - y1
        window_dim = max(h, w)
        center_x = int(x1 + round(w / 2))
        center_y = int(y1 + round(h / 2))

        for scale in scales:
            r_count += 1
            dim = window_dim * scale
            dim_half = round(dim / 2)
            dim = round(dim)
            x1 = center_x - dim_half
            y1 = center_y - dim_half
            x2 = center_x + dim_half
            y2 = center_y + dim_half

            # pre-process the region and scale it to img_dim
            region = img[y1:y2, x1:x2]
            region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
            region = skimage.exposure.equalize_hist(region)

            # we only need to store the region, it's top-left corner and sliding window dim
            regions.append(region)
            locations.append([x1, y1])
            window_dims.append(dim)

    regions = numpy.asarray(regions)
    locations = numpy.asarray(locations)
    window_dims = numpy.asarray(window_dims)

    ##############################
    # Start detection            #
    ##############################

    # split it to batches first, zero-pad them if needed
    regions = numpy.asarray(regions)
    n_regions = regions.shape[0]
    if n_regions % batch_size != 0:
        n_remaining = batch_size - (n_regions % batch_size)
        regions_padding = numpy.zeros(shape=(n_remaining, img_dim, img_dim), dtype=float)
        regions = numpy.vstack((regions, regions_padding))

    # run the detector on the regions
    start_time = time.clock()

    # loop on the batches of the regions
    n_batches = int(regions.shape[0] / batch_size)
    layer0_img_shape = (batch_size, 1, img_dim, img_dim)
    predictions = []
    for i in range(n_batches):
        # prediction: CNN filtering then MLP regression
        t1 = time.clock()
        batch = regions[i * batch_size: (i + 1) * batch_size]
        batch = batch.reshape(layer0_img_shape)
        filters = net_cnn(batch)
        filters = filters.reshape(net_mlp_input_shape).astype("float32")
        batch_pred = net_mlp.predict(filters)
        predictions.append(batch_pred)
        t2 = time.clock()
        print("... batch: %i/%i, time(sec.): %f" % ((i + 1), n_batches, t2 - t1))

    # after getting all the predictions, remove the padding
    predictions = numpy.hstack(predictions)[0:n_regions]

    end_time = time.clock()
    duration = (end_time - start_time) / 60.0
    print("... detection regions: %d, duration(min.): %f" % (r_count, duration))

    # construct the probability map for each scale and show it/ save it
    s_count = 0
    overlap_thresh = 0.5
    min_overlap = 0
    strong_prob_regions = []
    weak_prob_regions = []
    for pred, loc, window_dim in zip(predictions, locations, window_dims):
        s_count += 1
        w_regions, s_regions = __probability_map([pred], [loc], window_dim, overlap_thresh, min_overlap)
        if len(w_regions) > 0:
            weak_prob_regions.append(w_regions)
        if len(s_regions) > 0:
            strong_prob_regions.append(s_regions)
            print("Scale: %d, window_dim: %d, regions: %d, strong regions detected" % (s_count, window_dim, r_count))
        else:
            print("Scale: %d, window_dim: %d, regions: %d, no regions detected" % (s_count, window_dim, r_count))

    if len(weak_prob_regions) > 0:
        weak_prob_regions = numpy.vstack(weak_prob_regions)

    if len(strong_prob_regions) > 0:
        strong_prob_regions = numpy.vstack(strong_prob_regions)

    # now, after we finished scanning at all the levels, we should make the final verdict
    # by suppressing all the strong_regions that we extracted on different scales
    if len(strong_prob_regions) > 0:
        overlap_thresh = 0.25
        min_overlap = round(len(scales) * 0.35)
        weak_regions, strong_regions = CNN.nms.suppression(strong_prob_regions, overlap_thresh, min_overlap)
        return strong_regions, weak_regions, strong_prob_regions, weak_prob_regions
    else:
        return [], [], [], []


def __probability_map(predictions, locations, window_dim, overlap_thresh, min_overlap):
    locations = numpy.asarray(locations)
    predictions = numpy.asarray(predictions)

    regions = []
    idx = numpy.where(predictions)[0]
    for i in idx:
        region = [0, 0, window_dim, window_dim]
        location = locations[i]
        x1 = int(region[0] + location[0])
        y1 = int(region[1] + location[1])
        x2 = int(region[2] + location[0])
        y2 = int(region[3] + location[1])
        regions.append([x1, y1, x2, y2])

    # check if no region found
    if len(regions) == 0:
        return [], []

    # suppress the new regions and raw them with red color
    weak_regions, strong_regions = CNN.nms.suppression(regions, overlap_thresh, min_overlap)

    # return the map to be exploited later by the detector, for the next scale
    return weak_regions, strong_regions


def __save_detection_result(img_color, regions):
    strong_regions = regions[0]
    weak_regions = regions[1]
    strong_probability_regions = regions[2]
    weak_probability_regions = regions[3]

    # draw the result of the detection
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    green_color = (0, 255, 0)
    yellow_color = (84, 212, 255)
    for reg in weak_probability_regions:
        cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), green_color, 1)
    for reg in strong_probability_regions:
        cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), blue_color, 1)
    for reg in weak_regions:
        cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), yellow_color, 1)
    for reg in strong_regions:
        cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), red_color, 2)

    cv2.imwrite("D://_Dataset//GTSDB//Test_Regions/result.png", img_color)


# endregion

# region Classifier

def __build_classifier(model_path):
    # load the model
    with open(model_path, 'rb') as f:
        net_cnn = pickle.load(f)

    return net_cnn


def __classify(net_cnn, img, img_dim):
    img = img.reshape((1, 1, img_dim, img_dim))
    prediction = net_cnn.predict(img)
    return prediction


# endregion

# region View/Show/Plot


def __show_street_view_images(directions, api_key):
    # loop on all the points and get the google street view image at each one
    plt.figure(num=1, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='w')
    plt.ion()
    plt.axis('off')
    plt.show()

    # download the images of google street view at each location/step
    img_count = 0
    for dir in directions:
        latlng = "%f,%f" % (dir["lat"], dir["lng"])
        heading = dir["heading"]
        url = "https://maps.googleapis.com/maps/api/streetview?size=640x400&location=%s&heading=%f&pitch=0&key=%s" % (latlng, heading, api_key)
        img_bytes = requests.get(url).content
        img = numpy.asarray(PIL.Image.open(io.BytesIO(img_bytes)))
        plt.imshow(img)
        plt.pause(0.1)
        img_count += 1
        print("... new image: %d" % (img_count))


def __plot_points_on_map(locations, is_locations=True):
    if is_locations:
        points = __convert_locations_to_points(locations)
    else:
        points = locations
    zoom_level = 16
    p_count = len(points)
    center = points[int(p_count / 2)]
    mymap = pygmaps.pygmaps.maps(center[0], center[1], zoom_level)

    # mymap.setgrids(37.42, 37.43, 0.001, -122.15, -122.14, 0.001)
    # mymap.addradpoint(37.429, -122.145, 95, "#FF0000")

    # create range of colors for the points
    hex_colors = []
    for val in range(1, p_count + 1):
        col = __pseudo_color(val, 0, p_count)
        hex_colors.append(__rgb_to_hex(col))

    # draw marks at the points
    p_count = 0
    for pnt, col in zip(points, hex_colors):
        p_count += 1
        mymap.addpoint(pnt[0], pnt[1], col, title=str(p_count))

    # draw path using the points then show the map
    path_color = "#0A6491"
    mymap.addpath(points, path_color)
    mymap.draw('mymap.draw.html')
    url = 'mymap.draw.html'
    webbrowser.open_new_tab(url)


def __draw_image(img, num):
    # plot original image and first and second components of output
    # plt.figure(num)
    # plt.gray()
    # plt.ion()
    # plt.axis('off')
    plt.imshow(img)
    plt.show()


# endregion

# region Path Calculation


def __recode_path(direction_steps, frames_per_meter=1):
    """
    Generate points between every 2 points in the given steps
    This is to enrich the points within the path
    :param direction_steps:
    :param frames_per_meter:
    :return:
    """

    locations = []
    for i in range(0, len(direction_steps) - 1):
        step = direction_steps[i]
        distance = step["distance"]["value"]
        frames = int(distance * frames_per_meter)
        point1 = step["start_location"]
        point2 = step["end_location"]
        lat1 = point1["lat"]
        lng1 = point1["lng"]
        lat2 = point2["lat"]
        lng2 = point2["lng"]
        interpolated = __interpolate_path(lat1, lng1, lat2, lng2, frames)
        locations.append(interpolated)

    # update the directions so that when at a waypoint you're looking
    # towards the next
    locations = numpy.hstack(locations)
    locations = __calculate_heading(locations)

    return locations


def __adjust_pace(locations):
    """
    For the given locations, check if the distance between each one is near to the give
    pace, if not either add or remove locations to adjust the pace
    After that, calculate the heading of each point such that each one
    is pointing at/looking towards it's next
    :param locations:
    :return:
    """

    adjusted_locations = []
    return adjusted_locations


def __interpolate_path(lat1, lng1, lat2, lng2, frames):
    """
    Generate points between the points of the given start/stop points.
    :param lat1:
    :param lng1:
    :param lat2:
    :param lng2:
    :param frames:
    :return:
    """

    x = [lat1, lat2]
    y = [lng1, lng2]
    xvals = numpy.linspace(lat1, lat2, frames)
    yinterp = numpy.interp(xvals, x, y)

    # create geo location point with each point
    # as dictionary containing lat and lng values
    points = []
    for lat, lng in zip(xvals, yinterp):
        point = {"lat": lat, "lng": lng}
        points.append(point)

    return points


def __calculate_heading(locations):
    """
    For the given list of locations, calculate and add the heading for each of them
    :param locations:
    :return:
    """

    heading = 0
    n = len(locations)
    for i in range(0, n):
        loc = locations[i]
        if i < n - 1:
            heading = __compute_direction(loc, locations[i + 1])
        loc["heading"] = heading

    return locations


def __compute_direction(point1, point2):
    lat1 = point1["lat"]
    lng1 = point1["lng"]
    lat2 = point2["lat"]
    lng2 = point2["lng"]
    lambda1 = math.radians(lng1)
    lambda2 = math.radians(lng2)
    psi1 = math.radians(lat1)
    psi2 = math.radians(lat2)

    y = math.sin(lambda2 - lambda1) * math.cos(psi2)
    x = math.cos(psi1) * math.sin(psi2) - math.sin(psi1) * math.cos(psi2) * math.cos(lambda2 - lambda1)
    return math.degrees(math.atan2(y, x))


def __snap_result_to_locations(snap_result):
    locations = []
    for snap in snap_result:
        snap = snap["location"]
        loc = {"lat": snap["latitude"], "lng": snap["longitude"]}
        locations.append(loc)
    return locations


# endregion

# region Conversions


def __convert_steps_to_locations(steps):
    locations = []
    for step in steps:
        loc = step["start_location"]
        locations.append(loc)
    return locations


def __convert_locations_to_points(locations):
    points = []
    for loc in locations:
        points.append((loc["lat"], loc["lng"]))
    return points


def __convert_points_to_locations(points):
    locations = []
    for p in points:
        location = {"lat": p[0], "lng": p[1]}
        locations.append(location)
    return locations


# endregions

# region Color Manipluation

def __pseudo_color(val, minval, maxval):
    # convert val in range minval..maxval to the range 0..120 degrees which
    # correspond to the colors red..green in the HSV colorspace
    h = (float(val - minval) / (maxval - minval)) * 120
    # convert hsv color (h,1,1) to its rgb equivalent
    # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = colorsys.hsv_to_rgb(h / 360, 1., 1.)
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return r, g, b


def __hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def __rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


# endregion

# region Mics

def __read_api_key():
    file_path = "C://Users//Noureldien//Documents//PycharmProjects//TrafficSignRecognition//Data//google-maps-key.pkl"
    key = pickle.load(open(file_path, "rb"))
    return key


def __tutorial():
    api_key = __read_api_key()
    client = googlemaps.client.Client(key=api_key)

    # Geocoding and address
    geocode_result = googlemaps.client.geocode(client, '1600 Amphitheatre Parkway, Mountain View, CA')

    # Look up an address with reverse geocoding
    reverse_geocode_result = googlemaps.client.reverse_geocode(client, (40.714224, -73.961452))

    # Request directions via public transit
    now = googlemaps.client.datetime.now()
    directions_result = googlemaps.client.directions(client, "Sydney Town Hall", "Parramatta, NSW", mode="transit", departure_time=now)

# endregion
