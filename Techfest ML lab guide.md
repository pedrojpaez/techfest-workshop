---


---

<h1 id="machine-learning-in-action">Machine Learning in Action</h1>
<p>Author: Pedro Paez<br>
github: <a href="https://github.com/pedrojpaez/sagemaker-labs">https://github.com/pedrojpaez/techfest-workshop.git</a></p>
<p>In this lab we will be going through the entire Data Science workflow using <strong>Sagemaker</strong>. The objective of this exercise is to build from scratch a Data Science project and to learn how <strong>Sagemaker</strong> helps accelerate the process of building and deploying in production custom machine learning models. We will see how to leverage Sagemaker’s first party algorithms as well as the high level SDF for Deep Learning frameworks.</p>
<p>We will be building and end-to-end Natural Language Processing pipeline to classify newspaper headlines into general categories. We will first build word embeddings (vector representations of the english vocabulary) to enrich our model.</p>
<h2 id="prerequisites">Prerequisites:</h2>
<p>For this lab you will need to have:</p>
<ul>
<li>A laptop</li>
<li>Network connectivity</li>
<li>An AWS account</li>
<li>Basic Python scripting experience</li>
<li>Basic knowledge of Data Science workflow</li>
</ul>
<p>Preferred knowledge:</p>
<ul>
<li>Basic knowledge of containers</li>
<li>Basic knowledge of deep learning</li>
</ul>
<h2 id="part-1--prepare-environment-and-create-new-sagemaker-project">Part 1 : Prepare environment and create new Sagemaker project</h2>
<ol>
<li>Go to AWS Console in your account</li>
<li>On the top right corner select region N.Virginia</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/v-G3E9ggNbbeHRIXlEfxd4kXIq4aoHDD26hN6sJomu0WCuULD9XIWKCXc0gLWVuMohKbftRP2REK" alt="enter image description here" title="Console"></p>
<ol start="3">
<li>Search and click on Amazon Sagemaker</li>
<li>Under Notebook &gt; Select Notebook Instance &gt; and click on “Create Notebook Instance” button (orange button)</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/fpI3YGLlxKkbX05p6-tqgEn9PPorKD-r3VoA9EMWf0vemlzwJVTSb4N3lIVggye8OH6BFvB0Ctce" alt="enter image description here" title="Sagemaker console"></p>
<ol start="5">
<li>Give your project a name under “Notebook instance name”</li>
<li>Select ml.t2.medium Notebook instance type</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/OVLOWKUhRCVNgU8APDhzMIYJOSGiveN3vWenyUQbV0Yezo8SLBh3DZxoFN4_hpHdv3vme6TGoIvv" alt="enter image description here" title="Create Notebook"></p>
<ol start="6">
<li>
<p>Under “Permissions and encryption” &gt; Under IAM role &gt; select “Create a new role” in the scroll down menu<br>
<img src="https://lh3.googleusercontent.com/pcQEJ0PHR0Qr6yiksw-5RA2vepl5jvyOqj2hUW0SVMD-4tW82LN1LAlyX2Mdxl0EA3S0TR5zunn3" alt="enter image description here" title="Create Role"></p>
</li>
<li>
<p>Select “Any S3 bucket” &gt; Click on “Create new role” button</p>
</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/nVB4ET4mTMpVvK7VDdQF2sWJGSIK6ZMWC4uuPh1o2ckvBRSY0ualc23fFjJZjuJeYaZy_AcAtr8n" alt="enter image description here" title="Permissions"></p>
<ol start="8">
<li>Finally “Create Notebook Instance” and wait until status is “InService”</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/ufXQaaH8u53U-UDV8zofWT8choIq5Or6LeGYsVVmvoYU_ZJPp9Q2OBVrm_ifMwmK36y3qk6WF_kb" alt="enter image description here" title="InService"></p>
<h3 id="clone-git-repo-with-workshop-material">Clone git repo with workshop material</h3>
<ol>
<li>Select “Open Jupyter”. You should see a Jupyter notebook web interface.</li>
<li>Select “New” in the top right corner &gt; Click on “Terminal”. A new tab will open with access to the Shell.</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/4iSVe9Mx3jpVKZ90ATCatzGp11HQPz35l1z3f2oD7aJlt_3cwDrcySt9nZU9_WCtlQ8Wy0QFYgnt" alt="enter image description here" title="New"></p>
<ol>
<li>
<p>You now have shell access to the notebook instance and full control/flexibility over your environment. We will cd (change directory to the Sagemaker home directory). Type from the root directory : <code>cd Sagemaker</code></p>
</li>
<li>
<p>We will clone the material for this lab from the git repo : <a href="https://github.com/pedrojpaez/techfest-workshop.git">https://github.com/pedrojpaez/techfest-workshop.git</a></p>
<p>git clone <a href="https://github.com/pedrojpaez/techfest-workshop.git">https://github.com/pedrojpaez/techfest-workshop.git</a></p>
</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/uv6VViWYBUHjXwV0HPhI1cMO_PBs1ZnUSTyafDoRJVnIgepiguyQzkl6-pwQmMwOJVtFwXtguSID" alt="enter image description here" title="Shell"></p>
<ol start="3">
<li>Return to previous tab (Jupyter notebook web interface). The techfest-workshop directory should now be available.</li>
</ol>
<p><img src="https://lh3.googleusercontent.com/GwuHsSGFoTBQAmMT5IwQBhBL8M3fS8a90K3ChngNO_gTWj37YO8XuuVe7dDLtkHnY5_ockSCxHD6" alt="enter image description here" title="techfest directory"></p>
<h3 id="techfest-workshop-directory">techfest-workshop directory</h3>
<p>There are 4 elements in the techfest-workshop directory:</p>
<ul>
<li><strong>tf-src</strong>: This directory contains the MXNet training script for our  document classifier.</li>
<li><strong>blazingtext_word2vec_text8.ipynb</strong>: Notebook to create word embeddings using the Sagemaker first party algorithm Blazingtext. We will use these embeddings as input for our headline classifier to enrich the model.</li>
<li><strong>headline-classifier-local.ipynb</strong>: Notebook to create headline classifier using keras (with MXNet backend) on the local instance.</li>
<li><strong>headline-classifier-mxnet.ipynb</strong>: Notebook to create headline classifier leveraging Sagemaker training and deploying features. We will use MXNet high-level SDK to bring our MXNet code and run and deploy our model.</li>
</ul>
<p><img src="https://lh3.googleusercontent.com/nfOKIeWpM7X4Ae16r7xaR7RZCdRAkCqusvSLaKHmKOvS9sNJAy8CLVSmDHrjEI3CS7EjnqAR_aVA" alt="enter image description here"></p>
<h3 id="run-blazingtext_word2vec_text8.ipynb-notebook">Run blazingtext_word2vec_text8.ipynb notebook</h3>
<p>In this notebook we will run through the snippets of code. We will be building a word embedding model (vector representations of the english vocabulary) to use as input for our document classification model.</p>
<p>For this notebook we will use the first party algorithm Blazingtext to build our word embeddings and we will leverage the one-click training/one-click deployment capabilities of Sagemaker.</p>
<p>The general actions we will be running:</p>
<ol>
<li>Configure notebook</li>
<li>Download text8 corpus file</li>
<li>Upload data to S3</li>
<li>Run training job on Sagemaker</li>
<li>Deploy model</li>
<li>Download model object and unpack wordvectors</li>
<li>Clean up (delete model endpoint)</li>
</ol>
<p>Run through the notebook and read the instructions.</p>
<h3 id="run-headline-classifier-local.ipynb-notebook">Run headline-classifier-local.ipynb notebook</h3>
<p>In this notebook we will run through the snippets of code. We will build a headline classifier model that will classify newspaper headlines into 4 classes. We will build a deep learning model using the Keras interface with MXNet backend (and use the word embeddings we previously built as input to our model). We will run the training on locally (on the notebook instance) to evaluate performance.</p>
<p>The general actions we will be running:</p>
<ol>
<li>Configure notebook</li>
<li>Download NewsAggregator datasets</li>
<li>Upload data to S3</li>
<li>Run training job locally</li>
<li>Move to the next notebook.</li>
</ol>
<p>Run through the notebook and read the instructions.</p>
<h3 id="run-headline-classifier-mxnet.ipynb-notebook">Run headline-classifier-mxnet.ipynb notebook</h3>
<p>In this notebook we will run through the snippets of code. We will build a headline classifier model that will classify newspaper headlines into 4 classes. We will build a deep learning model using the Keras interface with MXNet backend (and use the word embeddings we previously built as input to our model). We will run the training on Sagemaker and package the MXNet code to a training script and we will evaluate performance. Finally we will deploy our model as a RESTful API.</p>
<p>The general actions we will be running:</p>
<ol>
<li>Configure notebook</li>
<li>Upload data to S3</li>
<li>Run training job on Sagemaker</li>
<li>Deploy model on Sagemaker</li>
<li>Clean up (delete model endpoint)</li>
</ol>
<p>Run through the notebook and read the instructions.</p>

