import pickle
import math
import numpy
import requests
import io

import PIL
import PIL.Image

import colorsys
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

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

    dummy = 10


def __draw_image(img, num):
    # plot original image and first and second components of output
    # plt.figure(num)
    # plt.gray()
    # plt.ion()
    # plt.axis('off')
    plt.imshow(img)
    plt.show()


def __read_api_key():
    file_path = "C://Users//Noureldien//Documents//PycharmProjects//TrafficSignRecognition//Data//google-maps-key.pkl"
    key = pickle.load(open(file_path, "rb"))
    return key


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


def __snap_result_to_locations(snap_result):
    locations = []
    for snap in snap_result:
        snap = snap["location"]
        loc = {"lat": snap["latitude"], "lng": snap["longitude"]}
        locations.append(loc)
    return locations


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


def __tutorial():
    api_key = __read_api_key()
    client = googlemaps.client.Client(key=api_key)

    # Geocoding and address
    geocode_result = googlemaps.client.geocode(client, '1600 Amphitheatre Parkway, Mountain View, CA')

    # Look up an address with reverse geocoding
    reverse_geocode_result = googlemaps.client.reverse_geocode(client, (40.714224, -73.961452))

    # Request directions via public transit
    now = googlemaps.client.datetime.now()
    directions_result = googlemaps.client.directions(client, "Sydney Town Hall",
                                                     "Parramatta, NSW",
                                                     mode="transit",
                                                     departure_time=now)
