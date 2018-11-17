Test
----

Finally, we can use Tester to calculate the accuracy of the model on the test set. In order to do so we must indicate the checkpoint containing the model that we want to test::


    tomaszs-mbp:mi-prometheus tomaszkornuta$ mip-tester --m ./experiments/MNIST/LeNet5/20181116_160127/models/model_best.pt
    Info: Parsing the /Users/tomaszkornuta/Documents/GitHub/mi-prometheus/experiments/MNIST/LeNet5/20181116_160127/training_configuration.yaml configuration file
    Loaded configuration from file /Users/tomaszkornuta/Documents/GitHub/mi-prometheus/experiments/MNIST/LeNet5/20181116_160127/training_configuration.yaml
    [2018-11-16 16:49:30] - INFO - Tester >>> Setting numpy random seed in testing to: 3971669138
    [2018-11-16 16:49:30] - INFO - Tester >>> Setting torch random seed in testing to: 2013212003
    [2018-11-16 16:49:30] - WARNING - Tester >>> GPU flag is disabled, using CPU.

Analogically to trainer, Tester will next create dataloader, problem and sampler:

    [2018-11-16 16:49:30] - INFO - ProblemFactory >>> Loading the MNIST problem from miprometheus.problems.image_to_class.mnist
    [2018-11-16 16:49:30] - WARNING - MNIST >>> Upscaling the images to [32, 32]. Slows down batch generation.
    [2018-11-16 16:49:30] - INFO - SamplerFactory >>> The sampler configuration section is not present.
    [2018-11-16 16:49:30] - INFO - Tester >>> Problem for 'testing' loaded (size: 10000)
    [2018-11-16 16:49:30] - INFO - Tester >>> Setting the max number of episodes to: 157

And model. Please notice that this time it has also loaded model parameters from checkpoint 

    [2018-11-16 16:49:30] - INFO - ModelFactory >>> Loading the LeNet5 model from miprometheus.models.vision.lenet5
    [2018-11-16 16:49:30] - WARNING - Model >>> No parameter value was parsed from problem_default_values_
    [2018-11-16 16:49:30] - INFO - Model >>> Imported LeNet5 parameters from checkpoint from 2018-11-16 16:04:28.446649 (episode: 5159, loss: 0.05859316140413284, status: Not converged: Epoch Limit reached)
    [2018-11-16 16:49:30] - INFO - Tester >>>


..note::
    Trainer has the same capability.
    If you want to load and e.g. finetune your model use option --m(odel).

After the setup phase, tester starts its experiment::

    [2018-11-16 16:49:30] - INFO - Tester >>> Testing over the entire test set (10000 samples in 157 episodes)
    [2018-11-16 16:49:30] - INFO - Tester >>> loss 0.0415740199; episode 000000; acc 0.9843750000; batch_size 000064 [Partial Test]
    [2018-11-16 16:49:31] - INFO - Tester >>> loss 0.0111131836; episode 000100; acc 1.0000000000; batch_size 000064 [Partial Test]
    [2018-11-16 16:49:32] - INFO - Tester >>>
    ================================================================================
    [2018-11-16 16:49:32] - INFO - Tester >>> Test finished
    [2018-11-16 16:49:32] - INFO - Tester >>> episode 000157; episodes_aggregated 000157; loss 0.0711115003; loss_min 0.0005262249; loss_max 0.6216378212; loss_std 0.0811631531; acc 0.9837778807; acc_min 0.9218750000; acc_max 1.0000000000; acc_std 0.0148953591; samples_aggregated 010000 [Full Test]