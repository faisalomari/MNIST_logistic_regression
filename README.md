<title>MNIST Classification with Logistic Regression and HOG Features</title>
<h1>MNIST Classification with Logistic Regression and HOG Features</h1>

<p>
    This repository contains code for training and evaluating a logistic regression model on the MNIST dataset using Histogram of Oriented Gradients (HOG) features. The HOG features are extracted from the images to capture the local structure and shape information, which are then used as input to the logistic regression model for classification.
</p>

<h2>Dataset</h2>

<p>
    The MNIST dataset is a widely used benchmark dataset in the field of machine learning. It consists of grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. The dataset is split into a training set and a test set, with corresponding labels for each image.
</p>

<h2>Dependencies</h2>

<p>
    The code requires the following dependencies:
</p>

<ul>
    <li>Python 3.x</li>
    <li>NumPy</li>
    <li>OpenCV (for HOG feature extraction)</li>
    <li>scikit-learn (for logistic regression model)</li>
</ul>

<p>
    You can install the required dependencies by running the following command:
</p>

<pre><code>pip install -r requirements.txt</code></pre>

<h2>Usage</h2>


<li>Clone the repository:
    <pre><code>git clone <a href="https://github.com/faisalomari/MNIST_logistic_regression">https://github.com/your-username/mnist-logistic-hog.git</a></code></pre>
</li>

<li>Just run the python notebook.
</li>



<h2>Results</h2>

<p>The trained logistic regression model achieves an accuracy of 95% on the test set, demonstrating the effectiveness of using HOG features for classifying MNIST digits.</p>
