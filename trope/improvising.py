#!/usr/bin/python3
import random
from itertools import product
import numpy as np
from scipy import stats
from scipy.spatial import distance


class Improv:

    def __init__(self, input): # Improv should only be taking a Performer; but it might be nicer to literally treat this as a general method :thonk: -- i.e. should this just work on random lists outside the context of trope? eh, I could always do that later-ish.
        if hasattr(input, 'refrain'): # durations are created when a Performer object is instantiated, regardless of whether they're explicitly defined by the end-user; checking for 'refrain' should be sufficient
            self.input_refrain = input.refrain
            self.input_durations = input.durations
        else:
            self.input = input


    def permutation(self):

        return np.random.permutation(self.input)


    def markov(self, walk_length=None):

        def _get_next(first, samples):
            next = np.nonzero(np.isin(samples, first))[0]
            if first == samples[-1]:
                next = next[:-1]
            return next + 1

        def _create_markov_dict(samples):
            markov_dict = {sample: np.asarray(samples)[_get_next(sample, samples)] for sample in set(samples)}
            return markov_dict

        def _create_markov_iteration(samples):
            output_len = len(samples) if walk_length is None else walk_length
            mode = stats.mode(samples)

            markov_iteration = []
            starting_point = samples[0]

            markov_dict = _create_markov_dict(samples)

            first_random = random.choice(markov_dict[starting_point])
            markov_iteration.extend([starting_point, first_random])

            while len(markov_iteration) < output_len:
                try:
                    markov_iteration.append(random.choice(markov_dict[markov_iteration[-1]]))
                except (IndexError, TypeError):
                    markov_iteration.append(mode[0][0])

            return markov_iteration
        
        # if the shape of `refrain` is > 1, create a markov iteration for each element
        refrain_shape = self.input.shape
        if len(refrain_shape) > 1:
            output_if_multidimensional = []
            for i in self.input:
                output_if_multidimensional.append(_create_markov_iteration(i))
            return np.asarray(output_if_multidimensional)

        return _create_markov_iteration(self.input)


    def rossmo(self):

        def normalize(data):
            return [(d - min(data)) / (max(data) - min(data)) for d in data]

        def invert_normalize(data, original_data):
            return [d * (max(original_data) - min(original_data)) + min(original_data) for d in data]

        # TODO see note in Performer about zipping refrain and durations earlier; that way I don't have to do that here and in Performer (and possibly elsewhere down the line)
        def normalize_inputs():

            adjusted_durations = [0] + self.input_durations[:-1]
            cummed_durations = np.cumsum(adjusted_durations)
            normalized_refrain = normalize(self.input_refrain)

            return list(zip(cummed_durations, normalized_refrain))

        xy = normalize_inputs()

        def get_max_distance(coordinates):
            '''
            Returns the maximum distance for coordinates.
            '''
            distances = distance.cdist(coordinates, coordinates)
            max_distance = np.max(distances)
            return max_distance

        def get_area_of_interest_boundaries(coordinates):
            '''
            The area of interest taken by:
            - first getting the the min and max x and y for all coordinates,getting us the corresponding xs and ys for the uppermost,lowermost, rightmost, and leftmost points.
            - then, we "fan" out from these by getting the max possibledistance between all coordinates, get_max_distance()

            This assumes that, for example:
            If the maximum distance between the list of all coordinates is 100, then the subject "resides" within 100 of the uppermost, lowermost, rightmost, and leftmost points.

            PUT THIS INSIDE get_area_of_interest()
            '''

            x  = [x for x,y in coordinates]
            y = [y for x,y in coordinates]
            upper, lower, right, left = np.max(y), np.min(y), np.max(x), np.min(x)

            max_distance = get_max_distance(coordinates)

            y_min, y_max = (max(0, lower - max_distance), max(0, upper + max_distance))
            x_min, x_max = (max(0, left - max_distance), max(0, right + max_distance))

            boundaries = (y_min, y_max, x_min, x_max)

            return boundaries

        def get_area_of_interest(coordinates, accuracy=None):
            '''
            Should rename this points of interest, I think.
            Rename accuracy to something like granularity

            ¡¡¡¡¡TRY np.meshgrid()!!!!!
            '''
            if accuracy is None:
                accuracy = len(coordinates)
            y_min, y_max, x_min, x_max =get_area_of_interest_boundaries(coordinates)
            y_range = np.linspace(y_min, y_max, num=accuracy)
            x_range = np.linspace(x_min, x_max, num=accuracy)

            area_of_interest = product(x_range, y_range)

            return area_of_interest

        def get_buffer(coordinates):
            '''
            Accepts a list of x- and y-coordinates.
            '''
            manhattan = distance.cdist(coordinates, coordinates,metric='cityblock')
            buffer = np.median(manhattan)

            return buffer

        def get_phi(manhattan_distance, buffer): # should probably be a lambda function
            return 1 if manhattan_distance > buffer else 0

        def rossmo_formula(coordinates, f=0.5, g=1):
            '''
            • area_of_interest: lat, lon coordinates for which we're trying to get probabilty of "residence"
            • f & g: The main idea of the formula is that the probability ofcrimes first increases as one moves through the buffer zone awayfrom the hotzone, but decreases afterwards. The variable f can bechosen so that it works best on data of past crimes. The sameidea goes for the variable g.

            get_phi() should probably be a lambda function
            '''
            area_of_interest = get_area_of_interest(coordinates)
            rossmo = {}
            B = get_buffer(coordinates)
            p = 0
            for a,i in area_of_interest:
                for x,y in coordinates:
                    manhattan = distance.cityblock([a, i],[x, y])
                    phi = get_phi(manhattan_distance=manhattan, buffer=B)
                    p += (phi / np.abs(manhattan) ** f) + ((1 - phi) * (B ** (g - f)) / ((2 * B) - np.abs(manhattan)) ** g)
                rossmo[(a,i)] = p
                p = 0

            return rossmo

        def get_coordinates_for_top_probabilities(coordinates, num=None):
            '''
            Returns the coordinates for the top n "probabilities of residence" for the subject.
            '''
            if num is None:
                num = len(coordinates)
            rossmo = rossmo_formula(coordinates)
            rossmo_values = rossmo.values()
            top_probabilities = sorted(rossmo_values)[-num:]
            top_points = [k for k,v in rossmo.items() if v in top_probabilities]

            return top_points

        def invert_normalize_output():
            top_scores = get_coordinates_for_top_probabilities(xy) # get top scoring coordinates
            x_out = [x for x,y in top_scores]
            y_out = invert_normalize([y for x,y in top_scores], self.input_refrain)
            return x_out, y_out # durations, refrain to be used - need to get the x_out figured out; note: np.diff doesn't get me what I initially thought because the output creates CHORDS in many cases, not a linear melody

        return invert_normalize_output()
