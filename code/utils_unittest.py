import utils
import unittest
import random


class TestUtils(unittest.TestCase):

    def test_calc_neuron(self):
        """
        Tests calc_neuron(t_inputs, t_weights, t_bias) function.

        The function calculates the output of one single neuron.
        The neuron:
        * takes each input, multiplies it by its weight
        * the bias is then added to the result
        """

        # test 100 times:
        for _ in range(100):
            rnd_value1 = random.uniform(-5000, 5000)
            rnd_value2 = random.uniform(-5000, 5000)
            rnd_value3 = random.uniform(-5000, 5000)
            rnd_value4 = random.uniform(-5000, 5000)
            rnd_value5 = random.uniform(-5000, 5000)
            rnd_value6 = random.uniform(-5000, 5000)

            rnd_bias = random.uniform(-5000, 5000)

            # ==============================================================
            # -- INPUTS SET TO ZERO
            # test case with 0 inputs, random weights, 0 bias
            result = utils.calc_neuron([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                    [rnd_value1, rnd_value2, rnd_value3,
                                    rnd_value4, rnd_value5, rnd_value6],
                                    0.0)
            
            self.assertEqual(result, 0.0)

            # test case with 0 inputs, random weights, random bias
            result = utils.calc_neuron([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                    [rnd_value1, rnd_value2, rnd_value3,
                                    rnd_value4, rnd_value5, rnd_value6],
                                    rnd_bias)
            
            self.assertEqual(result, rnd_bias)

            # ==============================================================
            # -- WEIGHTS SET TO ZERO

            # test case with random inputs, 0 weights, 0 bias
            result = utils.calc_neuron([rnd_value1, rnd_value2, rnd_value3,
                                    rnd_value4, rnd_value5, rnd_value6],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    0.0)
            
            self.assertEqual(result, 0.0)

            # test case with random inputs, 0 weights, random bias
            result = utils.calc_neuron([rnd_value1, rnd_value2, rnd_value3,
                                    rnd_value4, rnd_value5, rnd_value6],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    rnd_bias)
            
            self.assertEqual(result, rnd_bias)

            
            # ==============================================================
            # -- EXPECTED ERRORS

            # test case with more weights than inputs
            with self.assertRaises(ValueError):
                utils.calc_neuron([rnd_value1],
                                  [rnd_value2, rnd_value3],
                                  rnd_bias)
            
            # test case with more inputs than weights
            with self.assertRaises(ValueError):
                utils.calc_neuron([rnd_value4, rnd_value2, rnd_value3],
                                  [rnd_value5, rnd_value4],
                                  rnd_bias)
            
            # ==============================================================
            # -- EXPECTED RESULTS

            result = utils.calc_neuron([1, 2, 3], [0.2, 0.5, 1.7], 3)
            self.assertEqual(result, 9.3)
    
    def test_calc_layer_by_neurons(self):
        """
        Tests calc_layer_by_neurons(t_inputs, t_neurons) function.

        Calculates the output of a layer (outputs of all the neurons of the layer):
        * t_inputs: input data of every neuron
        * t_neurons: list of neurons properties (one dict with weights and bias for each neuron)
        """
        # test 100 times:
        for _ in range(100):
            rnd_value1 = random.uniform(-5000, 5000)
            rnd_value2 = random.uniform(-5000, 5000)
            rnd_value3 = random.uniform(-5000, 5000)
            rnd_value4 = random.uniform(-5000, 5000)
            rnd_value5 = random.uniform(-5000, 5000)
            rnd_value6 = random.uniform(-5000, 5000)

            rnd_bias = random.uniform(-5000, 5000)

            # ==============================================================
            # -- INPUTS SET TO ZERO
            # test case with 0 inputs, random weights, 0 bias for 5 neurons
            result = utils.calc_layer_by_neurons([0, 0, 0], [
                {"weights": [rnd_value1, rnd_value2, rnd_value3], "bias": 0},
                {"weights": [rnd_value4, rnd_value2, rnd_value3], "bias": 0},
                {"weights": [rnd_value5, rnd_value2, rnd_value3], "bias": 0},
                {"weights": [rnd_value1, rnd_value6, rnd_value2], "bias": 0},
                {"weights": [rnd_value1, rnd_value5, rnd_value3], "bias": 0},
                ])

            self.assertEqual(result, [0.0, 0.0, 0.0, 0.0, 0.0])

            # test case with 0 inputs, random weights, random bias for 5 neurons
            result = utils.calc_layer_by_neurons([0, 0, 0], [
                {"weights": [rnd_value1, rnd_value2, rnd_value3], "bias": rnd_bias},
                {"weights": [rnd_value4, rnd_value2, rnd_value3], "bias": rnd_bias},
                {"weights": [rnd_value5, rnd_value2, rnd_value3], "bias": rnd_bias},
                {"weights": [rnd_value1, rnd_value6, rnd_value2], "bias": rnd_bias},
                {"weights": [rnd_value1, rnd_value5, rnd_value3], "bias": rnd_bias},
                ])

            self.assertEqual(result, [rnd_bias, rnd_bias, rnd_bias, rnd_bias, rnd_bias])
    
    def test_calc_layer_by_props(self):
        """
        Tests calc_layer_by_props() function.
        """
        # TODO: test calc_layer_by_props()
        self.skipTest("TODO: test calc_layer_by_props()")

    def test_calc_batches_layer_by_props(self):
        """
        Tests calc_batches_layer_by_props() function.
        """
        # TODO: test calc_batches_layer_by_props()
        self.skipTest("TODO: test calc_batches_layer_by_props()")

    def test_neuron_relu_function(self):
        """
        Tests neuron_relu_function(t_input) function.

        The function should return zero if t_input is less than 0.
        Otherwise the return is t_input
        """
        # test many values
        for n in range(-100000, 100000):
            # calculate ReLU of n
            result = utils.neuron_relu_function(n)
            
            # result should be zero if n is less than 0.
            # Otherwise result is equal to n
            if n < 0:
                self.assertEqual(result, 0)
            else:
                self.assertEqual(result, n)

    def test_layer_relu_function(self):
        """
        Tests layer_relu_function() function.
        """
        # TODO: test layer_relu_function()
        self.skipTest("TODO: test layer_relu_function()")

    def test_batch_layer_relu_function(self):
        """
        Tests batch_layer_relu_function() function.
        """
        # TODO: test batch_layer_relu_function()
        self.skipTest("TODO: test batch_layer_relu_function()")


if __name__ == "__main__":

    unittest.main()