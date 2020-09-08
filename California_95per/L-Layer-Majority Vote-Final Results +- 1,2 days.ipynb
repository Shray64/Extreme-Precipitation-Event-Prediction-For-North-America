{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "nex_nw500_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_nex_nw500.txt\", header = None, delim_whitespace=True)\n",
    "nex_nu500_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_nex_nu500.txt\", header = None, delim_whitespace=True)\n",
    "nex_nv500_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_nex_nv500.txt\", header = None, delim_whitespace=True)\n",
    "nex_ntpw_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_nex_ntpw.txt\", header = None, delim_whitespace=True)\n",
    "nex_nqv2m_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_nex_nqv2m.txt\", header = None, delim_whitespace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "ex_nw500_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_ex_nw500.txt\", header = None, delim_whitespace=True)\n",
    "ex_nu500_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_ex_nu500.txt\", header = None, delim_whitespace=True)\n",
    "ex_nv500_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_ex_nv500.txt\", header = None, delim_whitespace=True)\n",
    "ex_ntpw_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_ex_ntpw.txt\", header = None, delim_whitespace=True)\n",
    "ex_nqv2m_7905 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_7905_ex_nqv2m.txt\", header = None, delim_whitespace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "indicator_0614 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_0614_indicator.txt\", header = None, delim_whitespace=True)\n",
    "test_set_y_initial = indicator_0614[4]\n",
    "test_set_y = test_set_y_initial.to_numpy().reshape(1, test_set_y_initial.shape[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "nw500_0614 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_0614_nw500.txt\", header = None, delim_whitespace=True)\n",
    "nu500_0614 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_0614_nu500.txt\", header = None, delim_whitespace=True)\n",
    "nv500_0614 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_0614_nv500.txt\", header = None, delim_whitespace=True)\n",
    "ntpw_0614 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_0614_ntpw.txt\", header = None, delim_whitespace=True)\n",
    "nqv2m_0614 = pd.read_csv(\"/Users/MyFolders/MIT/Datasets/DJF_0614_nqv2m.txt\", header = None, delim_whitespace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(Z):   \n",
    "    A = 1/(1+np.exp(-Z))\n",
    "    cache = Z\n",
    "    \n",
    "    return A, cache\n",
    "\n",
    "def relu(Z):    \n",
    "    A = np.maximum(0,Z)\n",
    "    \n",
    "    assert(A.shape == Z.shape)\n",
    "    \n",
    "    cache = Z \n",
    "    return A, cache"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def relu_backward(dA, cache):   \n",
    "    Z = cache\n",
    "    dZ = np.array(dA, copy=True) # just converting dz to a correct object.\n",
    "    \n",
    "    # When z <= 0, you should set dz to 0 as well. \n",
    "    dZ[Z <= 0] = 0\n",
    "    \n",
    "    assert (dZ.shape == Z.shape)\n",
    "    \n",
    "    return dZ\n",
    "\n",
    "def sigmoid_backward(dA, cache):    \n",
    "    Z = cache\n",
    "    \n",
    "    s = 1/(1+np.exp(-Z))\n",
    "    dZ = dA * s * (1-s)\n",
    "    \n",
    "    assert (dZ.shape == Z.shape)\n",
    "    \n",
    "    return dZ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def initialize_parameters_deep(layer_dims):   \n",
    "    np.random.seed(1)\n",
    "    parameters = {}\n",
    "    L = len(layer_dims)            # number of layers in the network\n",
    "\n",
    "    for l in range(1, L):\n",
    "        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01\n",
    "        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))\n",
    "        \n",
    "        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))\n",
    "        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))\n",
    "\n",
    "        \n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def linear_forward(A, W, b):\n",
    "    \n",
    "    Z = W.dot(A) + b\n",
    "    \n",
    "    assert(Z.shape == (W.shape[0], A.shape[1]))\n",
    "    cache = (A, W, b)\n",
    "    \n",
    "    return Z, cache"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def linear_activation_forward(A_prev, W, b, activation):   \n",
    "    if activation == \"sigmoid\":\n",
    "        # Inputs: \"A_prev, W, b\". Outputs: \"A, activation_cache\".\n",
    "        Z, linear_cache = linear_forward(A_prev, W, b)\n",
    "        A, activation_cache = sigmoid(Z)\n",
    "    \n",
    "    elif activation == \"relu\":\n",
    "        # Inputs: \"A_prev, W, b\". Outputs: \"A, activation_cache\".\n",
    "        Z, linear_cache = linear_forward(A_prev, W, b)\n",
    "        A, activation_cache = relu(Z)\n",
    "    \n",
    "    assert (A.shape == (W.shape[0], A_prev.shape[1]))\n",
    "    cache = (linear_cache, activation_cache)\n",
    "    \n",
    "    return A, cache\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def L_model_forward(X, parameters):\n",
    "    caches = []\n",
    "    A = X\n",
    "    L = len(parameters) // 2                  # number of layers in the neural network\n",
    "    \n",
    "    # Implement [LINEAR -> RELU]*(L-1). Add \"cache\" to the \"caches\" list.\n",
    "    for l in range(1, L):\n",
    "        A_prev = A \n",
    "        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = \"relu\")\n",
    "        caches.append(cache)\n",
    "    \n",
    "    # Implement LINEAR -> SIGMOID. Add \"cache\" to the \"caches\" list.\n",
    "    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = \"sigmoid\")\n",
    "    #print(AL)\n",
    "    caches.append(cache)\n",
    "    \n",
    "    assert(AL.shape == (1,X.shape[1]))\n",
    "            \n",
    "    return AL, caches"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_cost(AL, Y):   \n",
    "    m = Y.shape[1]\n",
    "\n",
    "    # Compute loss from aL and y.\n",
    "    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))\n",
    "    \n",
    "    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).\n",
    "    assert(cost.shape == ())\n",
    "    \n",
    "    return cost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def linear_backward(dZ, cache):\n",
    "    A_prev, W, b = cache\n",
    "    m = A_prev.shape[1]\n",
    "\n",
    "    dW = 1./m * np.dot(dZ,A_prev.T)\n",
    "    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)\n",
    "    dA_prev = np.dot(W.T,dZ)\n",
    "    \n",
    "    assert (dA_prev.shape == A_prev.shape)\n",
    "    assert (dW.shape == W.shape)\n",
    "    assert (db.shape == b.shape)\n",
    "    \n",
    "    return dA_prev, dW, db"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def linear_activation_backward(dA, cache, activation):\n",
    "    linear_cache, activation_cache = cache\n",
    "    \n",
    "    if activation == \"relu\":\n",
    "        dZ = relu_backward(dA, activation_cache)\n",
    "        dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
    "        \n",
    "    elif activation == \"sigmoid\":\n",
    "        dZ = sigmoid_backward(dA, activation_cache)\n",
    "        dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
    "    \n",
    "    return dA_prev, dW, db"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def L_model_backward(AL, Y, caches):\n",
    "    grads = {}\n",
    "    L = len(caches) # the number of layers\n",
    "    m = AL.shape[1]\n",
    "    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL\n",
    "    \n",
    "    # Initializing the backpropagation\n",
    "    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))\n",
    "    \n",
    "    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: \"AL, Y, caches\". Outputs: \"grads[\"dAL\"], grads[\"dWL\"], grads[\"dbL\"]\n",
    "    current_cache = caches[L-1]\n",
    "    grads[\"dA\" + str(L-1)], grads[\"dW\" + str(L)], grads[\"db\" + str(L)] = linear_activation_backward(dAL, current_cache, activation = \"sigmoid\")\n",
    "    \n",
    "    for l in reversed(range(L-1)):\n",
    "        # lth layer: (RELU -> LINEAR) gradients.\n",
    "        current_cache = caches[l]\n",
    "        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[\"dA\" + str(l + 1)], current_cache, activation = \"relu\")\n",
    "        grads[\"dA\" + str(l)] = dA_prev_temp\n",
    "        grads[\"dW\" + str(l + 1)] = dW_temp\n",
    "        grads[\"db\" + str(l + 1)] = db_temp\n",
    "\n",
    "    return grads"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def update_parameters(parameters, grads, learning_rate):   \n",
    "    L = len(parameters) // 2 # number of layers in the neural network\n",
    "\n",
    "    # Update rule for each parameter. Use a for loop.\n",
    "    for l in range(L):\n",
    "        parameters[\"W\" + str(l+1)] = parameters[\"W\" + str(l+1)] - learning_rate * grads[\"dW\" + str(l+1)]\n",
    "        parameters[\"b\" + str(l+1)] = parameters[\"b\" + str(l+1)] - learning_rate * grads[\"db\" + str(l+1)]\n",
    "        \n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_train_test(nex_names, ex_names, test_names):\n",
    "    nex_combo_7905 = pd.concat(nex_names, axis = 1)\n",
    "    nex_combo_7905[\"label\"] = 0\n",
    "\n",
    "    ex_combo_7905 = pd.concat(ex_names, axis = 1)\n",
    "    ex_combo_7905[\"label\"] = 1\n",
    "\n",
    "    combo_7905_df = pd.concat([ex_combo_7905, nex_combo_7905])\n",
    "    \n",
    "    train_combo_y_initial = combo_7905_df['label'].to_numpy()\n",
    "    train_combo_y = train_combo_y_initial.reshape(1, train_combo_y_initial.shape[0])\n",
    "\n",
    "    train_combo_x = combo_7905_df.iloc[:,0:combo_7905_df.shape[1]-1].to_numpy().T\n",
    "\n",
    "    test_df = pd.concat(test_names, axis = 1)\n",
    "\n",
    "    test_combo_x = test_df.to_numpy().T\n",
    "    \n",
    "    return train_combo_x, test_combo_x, train_combo_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: L_layer_model\n",
    "\n",
    "def L_layer_model(train_combo_x, train_combo_y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009\n",
    "\n",
    "    np.random.seed(1)\n",
    "    costs = []                         # keep track of cost\n",
    "    \n",
    "    # Parameters initialization. (≈ 1 line of code)\n",
    "    ### START CODE HERE ###\n",
    "    parameters = initialize_parameters_deep(layers_dims)\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    # Loop (gradient descent)\n",
    "    for i in range(0, num_iterations):\n",
    "\n",
    "        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.\n",
    "        ### START CODE HERE ### (≈ 1 line of code)\n",
    "        AL, caches = L_model_forward(train_combo_x, parameters)\n",
    "        ### END CODE HERE ###\n",
    "        \n",
    "        # Compute cost.\n",
    "        ### START CODE HERE ### (≈ 1 line of code)\n",
    "        cost = compute_cost(AL, train_combo_y)\n",
    "        ### END CODE HERE ###\n",
    "    \n",
    "        # Backward propagation.\n",
    "        ### START CODE HERE ### (≈ 1 line of code)\n",
    "        grads = L_model_backward(AL, train_combo_y, caches)\n",
    "        ### END CODE HERE ###\n",
    " \n",
    "        # Update parameters.\n",
    "        ### START CODE HERE ### (≈ 1 line of code)\n",
    "        parameters = update_parameters(parameters, grads, learning_rate)\n",
    "        ### END CODE HERE ###\n",
    "                \n",
    "        # Print the cost every 100 training example\n",
    "#         if print_cost and i % 100 == 0:\n",
    "#             print (\"Cost after iteration %i: %f\" %(i, cost))\n",
    "#         if print_cost and i % 100 == 0:\n",
    "#             costs.append(cost)\n",
    "            \n",
    "#     # plot the cost\n",
    "#     plt.plot(np.squeeze(costs))\n",
    "#     plt.ylabel('cost')\n",
    "#     plt.xlabel('iterations (per hundreds)')\n",
    "#     plt.title(\"Learning rate =\" + str(learning_rate))\n",
    "#     plt.show()\n",
    "    \n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def conf_matrix(predictions, y):\n",
    "    cm = np.zeros((2,2), dtype = int)\n",
    "    for i in range(y.shape[1]):\n",
    "        if(y[0,i] == 1 and predictions[0,i] == 1):\n",
    "            cm[0,0] += 1\n",
    "        if(y[0,i] == 1 and predictions[0,i] == 0):\n",
    "            cm[0,1] += 1\n",
    "        if(y[0,i] == 0 and predictions[0,i] == 1):\n",
    "            cm[1,0] += 1\n",
    "        if(y[0,i] == 0 and predictions[0,i] == 0):\n",
    "            cm[1,1] += 1\n",
    "#     cm[0,2] = np.sum(cm[0])\n",
    "#     cm[1,2] = np.sum(cm[1])\n",
    "#     cm[2,0] = np.sum(cm.T[0])\n",
    "#     cm[2,1] = np.sum(cm.T[1])\n",
    "#     cm[2,2] = np.sum(cm[2])\n",
    "    #print(cm)\n",
    "    return cm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def evalmetrics(cm):\n",
    "    tp = cm[0,0]\n",
    "    fn = cm[0,1]\n",
    "    fp = cm[1,0]\n",
    "    tn = cm[1,1]\n",
    "    \n",
    "    tpr = tp/(tp+fn)\n",
    "    fpr = fp/(fp+tn)\n",
    "    precision = tp/(tp+fp)\n",
    "    f1 = 2*precision*tpr/(precision + tpr)\n",
    "    \n",
    "    return tpr, fpr, f1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(X, y, parameters):\n",
    "    \n",
    "    m = X.shape[1]\n",
    "    n = len(parameters) // 2 # number of layers in the neural network\n",
    "    p = np.zeros((1,m), dtype = int)\n",
    "    j = 0\n",
    "    \n",
    "    # Forward propagation\n",
    "    probas, caches = L_model_forward(X, parameters)\n",
    "\n",
    "    \n",
    "    # convert probas to 0/1 predictions\n",
    "    for i in range(0, probas.shape[1]):\n",
    "        if probas[0,i] > 0.5:\n",
    "#             print(\"Count is %i and day number is %i\" %(j, i+1))\n",
    "#             j = j+1\n",
    "#             print(probas[0,i])\n",
    "#             if(y[0,i] == 0):\n",
    "#                 pred_extreme.append(i+1)\n",
    "            p[0,i] = 1\n",
    "        else:\n",
    "#             print(\"Count is %i and day number is %i\" %(j, i+1))\n",
    "#             j = j+1\n",
    "#             print(probas[0,i])\n",
    "            p[0,i] = -1\n",
    "    \n",
    "    #print results\n",
    "    #print (\"predictions: \" + str(p))\n",
    "    #print (\"true labels: \" + str(y))\n",
    "    #print(\"Accuracy: \"  + str(np.sum((p == y)/m)))\n",
    "        \n",
    "    return p"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# nw500, nqv2m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_combo_x, test_combo_x, train_combo_y = get_train_test([nex_nw500_7905,nex_nqv2m_7905], [ex_nw500_7905, ex_nqv2m_7905], [nw500_0614, nqv2m_0614])\n",
    "layer_dims = [train_combo_x.shape[0],2,2,2,1]\n",
    "parameters = L_layer_model(train_combo_x, train_combo_y, layer_dims, learning_rate = 0.5, num_iterations = 170, print_cost=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_test_1 = predict(test_combo_x, test_set_y, parameters)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_train_1 = predict(train_combo_x, train_combo_y, parameters)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# nu500, nv500, nqv2m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_combo_x, test_combo_x, train_combo_y = get_train_test([nex_nu500_7905, nex_nv500_7905, nex_nqv2m_7905], [ex_nu500_7905, ex_nv500_7905, ex_nqv2m_7905], [nu500_0614, nv500_0614, nqv2m_0614])\n",
    "layer_dims = [train_combo_x.shape[0],3,4,3,1]\n",
    "parameters = L_layer_model(train_combo_x, train_combo_y, layer_dims, learning_rate = 0.5, num_iterations = 180, print_cost=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_test_2 = predict(test_combo_x, test_set_y, parameters)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_train_2 = predict(train_combo_x, train_combo_y, parameters)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# nu500, nv500, nw500, nqv2m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_combo_x, test_combo_x, train_combo_y = get_train_test([nex_nu500_7905, nex_nv500_7905,nex_nw500_7905, nex_nqv2m_7905], [ex_nu500_7905, ex_nv500_7905, ex_nw500_7905, ex_nqv2m_7905], [nu500_0614,nv500_0614,nw500_0614, nqv2m_0614])\n",
    "layer_dims = [train_combo_x.shape[0],4,3,3,1]\n",
    "parameters = L_layer_model(train_combo_x, train_combo_y, layer_dims, learning_rate = 1.2, num_iterations = 479, print_cost=True)   \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_test_3 = predict(test_combo_x, test_set_y, parameters)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_train_3 = predict(train_combo_x, train_combo_y, parameters)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# nu500, nv500, nw500, ntpw"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_combo_x, test_combo_x, train_combo_y = get_train_test([nex_nu500_7905, nex_nv500_7905, nex_nw500_7905, nex_ntpw_7905], [ex_nu500_7905, ex_nv500_7905, ex_nw500_7905, ex_ntpw_7905], [nu500_0614, nv500_0614, nw500_0614, ntpw_0614])\n",
    "layer_dims = [train_combo_x.shape[0],2,3,2,1]\n",
    "parameters = L_layer_model(train_combo_x, train_combo_y, layer_dims, learning_rate = 0.5, num_iterations = 158, print_cost=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_test_4 = predict(test_combo_x, test_set_y, parameters)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_train_4 = predict(train_combo_x, train_combo_y, parameters)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# With the 3 special cases"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "majority_train = np.add(np.add(np.add(predictions_train_1, predictions_train_2), predictions_train_3), predictions_train_4)\n",
    "# majority_train[majority_train > 0] = 1\n",
    "# majority_train[majority_train <= 0] = 0\n",
    "\n",
    "for i in range(majority_train.shape[1]):\n",
    "    if majority_train[0,i] > 0:\n",
    "        if (predictions_train_1[0,i] == 1) and (predictions_train_2[0,i] == 1) and (predictions_train_3[0,i] == 1) and  (predictions_train_4[0,i] == -1):\n",
    "            majority_train[0,i] = 0\n",
    "        elif (predictions_train_1[0,i] == 1) and (predictions_train_2[0,i] == 1) and (predictions_train_3[0,i] == -1) and  (predictions_train_4[0,i] == 1):\n",
    "            majority_train[0,i] = 0\n",
    "        else:\n",
    "            majority_train[0,i] = 1\n",
    "   \n",
    "    else:\n",
    "        if (predictions_train_1[0,i] == 1) and (predictions_train_2[0,i] == -1) and (predictions_train_3[0,i] == -1) and  (predictions_train_4[0,i] == 1):\n",
    "            majority_train[0,i] = 1\n",
    "        else:\n",
    "            majority_train[0,i] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calibration statistics (1979 - 2005) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>163</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>42</td>\n",
       "      <td>2230</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                     163                      2\n",
       "actual_non-extreme (764)                 42                   2230"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.987879\n",
      "False positive rate: 0.018486\n",
      "F1 score: 0.881081\n"
     ]
    }
   ],
   "source": [
    "cm_train = conf_matrix(majority_train, train_combo_y)\n",
    "confusion_train = pd.DataFrame(cm_train, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Calibration statistics (1979 - 2005) :\")\n",
    "display(confusion_train)\n",
    "tpr, fpr, f1 = evalmetrics(cm_train)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [],
   "source": [
    "# predictions_test_3[predictions_test_3 == -3] = -1\n",
    "# predictions_test_4[predictions_test_4 == -3] = -1\n",
    "majority = np.add(np.add(np.add(predictions_test_1, predictions_test_2), predictions_test_3), predictions_test_4)\n",
    "# majority[majority > 0] = 1\n",
    "# majority[majority <= 0] = 0\n",
    "\n",
    "for i in range(majority.shape[1]):\n",
    "    if majority[0,i] > 0:\n",
    "        if (predictions_test_1[0,i] == 1) and (predictions_test_2[0,i] == 1) and (predictions_test_3[0,i] == 1) and  (predictions_test_4[0,i] == -1):\n",
    "            majority[0,i] = 0\n",
    "        elif (predictions_test_1[0,i] == 1) and (predictions_test_2[0,i] == 1) and (predictions_test_3[0,i] == -1) and  (predictions_test_4[0,i] == 1):\n",
    "            majority[0,i] = 0\n",
    "        else:\n",
    "            majority[0,i] = 1\n",
    "   \n",
    "    else:\n",
    "        if (predictions_test_1[0,i] == 1) and (predictions_test_2[0,i] == -1) and (predictions_test_3[0,i] == -1) and  (predictions_test_4[0,i] == 1):\n",
    "            majority[0,i] = 1\n",
    "        else:\n",
    "            majority[0,i] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation statistics (2006 - 2014) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>31</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>35</td>\n",
       "      <td>729</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                      31                     17\n",
       "actual_non-extreme (764)                 35                    729"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.645833\n",
      "False positive rate: 0.045812\n",
      "F1 score: 0.543860\n"
     ]
    }
   ],
   "source": [
    "cm_test = conf_matrix(majority, test_set_y)\n",
    "confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Validation statistics (2006 - 2014) :\")\n",
    "display(confusion_test)\n",
    "tpr, fpr, f1 = evalmetrics(cm_test)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [],
   "source": [
    "fpdays = []\n",
    "for i in range(test_set_y.shape[1]):\n",
    "    if majority[0,i] == 1: \n",
    "        if test_set_y[0,i] == 0:\n",
    "            fpdays.append(i)\n",
    "            #print(i+1, predictions_test_1[0][i], predictions_test_2[0][i],predictions_test_3[0][i],predictions_test_4[0][i])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# +-1 day"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [],
   "source": [
    "truefp1 = len(fpdays)\n",
    "j = 0\n",
    "for i in fpdays:\n",
    "    before = indicator_0614.iloc[i-1, 4]\n",
    "    actual = indicator_0614.iloc[i], 4\n",
    "    after = indicator_0614.iloc[i+1, 4]\n",
    "    \n",
    "    if (before == 1) or (after == 1):\n",
    "        #print(fpdays[j]+1)\n",
    "        truefp1 = truefp1 - 1\n",
    "    j = j+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation statistics (2006 - 2014) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>31</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>21</td>\n",
       "      <td>729</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                      31                     17\n",
       "actual_non-extreme (764)                 21                    729"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.645833\n",
      "False positive rate: 0.028000\n",
      "F1 score: 0.620000\n"
     ]
    }
   ],
   "source": [
    "cm_test[1,0] = truefp1\n",
    "confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Validation statistics (2006 - 2014) :\")\n",
    "display(confusion_test)\n",
    "tpr, fpr, f1 = evalmetrics(cm_test)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# +- 2 days"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [],
   "source": [
    "truefp2 = len(fpdays)\n",
    "j = 0\n",
    "for i in fpdays:\n",
    "    \n",
    "    before2 = indicator_0614.iloc[i-2, 4]\n",
    "    before1 = indicator_0614.iloc[i-1, 4]\n",
    "    actual = indicator_0614.iloc[i], 4\n",
    "    after1 = indicator_0614.iloc[i+1, 4]\n",
    "    after2 = indicator_0614.iloc[i+2, 4]\n",
    "    \n",
    "    if (before1 == 1) or (before2 == 1) or (after1 == 1) or (after2 == 1):\n",
    "        #print(fpdays[j] + 1)\n",
    "        truefp2 = truefp2 - 1\n",
    "    j = j+1\n",
    "\n",
    "#print(truefp2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation statistics (2006 - 2014) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>31</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>16</td>\n",
       "      <td>729</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                      31                     17\n",
       "actual_non-extreme (764)                 16                    729"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.645833\n",
      "False positive rate: 0.021477\n",
      "F1 score: 0.652632\n"
     ]
    }
   ],
   "source": [
    "cm_test[1,0] = truefp2\n",
    "confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Validation statistics (2006 - 2014) :\")\n",
    "display(confusion_test)\n",
    "tpr, fpr, f1 = evalmetrics(cm_test)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Without the 3 special cases"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [],
   "source": [
    "majority_train = np.add(np.add(np.add(predictions_train_1, predictions_train_2), predictions_train_3), predictions_train_4)\n",
    "majority_train[majority_train > 0] = 1\n",
    "majority_train[majority_train <= 0] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [],
   "source": [
    "majority = np.add(np.add(np.add(predictions_test_1, predictions_test_2), predictions_test_3), predictions_test_4)\n",
    "majority[majority > 0] = 1\n",
    "majority[majority <= 0] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Calibration statistics (1979 - 2005) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>165</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>60</td>\n",
       "      <td>2212</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                     165                      0\n",
       "actual_non-extreme (764)                 60                   2212"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 1.000000\n",
      "False positive rate: 0.026408\n",
      "F1 score: 0.846154\n"
     ]
    }
   ],
   "source": [
    "cm_train = conf_matrix(majority_train, train_combo_y)\n",
    "confusion_train = pd.DataFrame(cm_train, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Calibration statistics (1979 - 2005) :\")\n",
    "display(confusion_train)\n",
    "tpr, fpr, f1 = evalmetrics(cm_train)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation statistics (2006 - 2014) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>29</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>41</td>\n",
       "      <td>723</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                      29                     19\n",
       "actual_non-extreme (764)                 41                    723"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.604167\n",
      "False positive rate: 0.053665\n",
      "F1 score: 0.491525\n"
     ]
    }
   ],
   "source": [
    "cm_test = conf_matrix(majority, test_set_y)\n",
    "confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Validation statistics (2006 - 2014) :\")\n",
    "display(confusion_test)\n",
    "tpr, fpr, f1 = evalmetrics(cm_test)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [],
   "source": [
    "fpdays = []\n",
    "for i in range(test_set_y.shape[1]):\n",
    "    if majority[0,i] == 1: \n",
    "        if test_set_y[0,i] == 0:\n",
    "            fpdays.append(i)\n",
    "            #print(i+1, predictions_test_1[0][i], predictions_test_2[0][i],predictions_test_3[0][i],predictions_test_4[0][i])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# +- 1 day"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [],
   "source": [
    "truefp1 = len(fpdays)\n",
    "j = 0\n",
    "for i in fpdays:\n",
    "    before = indicator_0614.iloc[i-1, 4]\n",
    "    actual = indicator_0614.iloc[i], 4\n",
    "    after = indicator_0614.iloc[i+1, 4]\n",
    "    \n",
    "    if (before == 1) or (after == 1):\n",
    "        #print(fpdays[j]+1)\n",
    "        truefp1 = truefp1 - 1\n",
    "    j = j+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation statistics (2006 - 2014) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>29</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>24</td>\n",
       "      <td>723</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                      29                     19\n",
       "actual_non-extreme (764)                 24                    723"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.604167\n",
      "False positive rate: 0.032129\n",
      "F1 score: 0.574257\n"
     ]
    }
   ],
   "source": [
    "cm_test[1,0] = truefp1\n",
    "confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Validation statistics (2006 - 2014) :\")\n",
    "display(confusion_test)\n",
    "tpr, fpr, f1 = evalmetrics(cm_test)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# +- 2 days"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [],
   "source": [
    "truefp2 = len(fpdays)\n",
    "j = 0\n",
    "for i in fpdays:\n",
    "    \n",
    "    before2 = indicator_0614.iloc[i-2, 4]\n",
    "    before1 = indicator_0614.iloc[i-1, 4]\n",
    "    actual = indicator_0614.iloc[i], 4\n",
    "    after1 = indicator_0614.iloc[i+1, 4]\n",
    "    after2 = indicator_0614.iloc[i+2, 4]\n",
    "    \n",
    "    if (before1 == 1) or (before2 == 1) or (after1 == 1) or (after2 == 1):\n",
    "        #print(fpdays[j] + 1)\n",
    "        truefp2 = truefp2 - 1\n",
    "    j = j+1\n",
    "\n",
    "#print(truefp2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation statistics (2006 - 2014) :\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>predicted_extreme</th>\n",
       "      <th>predicted_non-extreme</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>actual_extreme (48)</th>\n",
       "      <td>29</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>actual_non-extreme (764)</th>\n",
       "      <td>18</td>\n",
       "      <td>723</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          predicted_extreme  predicted_non-extreme\n",
       "actual_extreme (48)                      29                     19\n",
       "actual_non-extreme (764)                 18                    723"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True positive rate: 0.604167\n",
      "False positive rate: 0.024291\n",
      "F1 score: 0.610526\n"
     ]
    }
   ],
   "source": [
    "cm_test[1,0] = truefp2\n",
    "confusion_test = pd.DataFrame(cm_test, index=['actual_extreme (48)', 'actual_non-extreme (764)'],\n",
    "                         columns=['predicted_extreme','predicted_non-extreme'])\n",
    "\n",
    "print(\"Validation statistics (2006 - 2014) :\")\n",
    "display(confusion_test)\n",
    "tpr, fpr, f1 = evalmetrics(cm_test)\n",
    "print(\"True positive rate: %f\" %(tpr))\n",
    "print(\"False positive rate: %f\" %(fpr))\n",
    "print(\"F1 score: %f\" %(f1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
