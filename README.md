<!DOCTYPE html>
<html>
<head>
  
</head>
<body>
  <h1>Medical Chatbot</h1>
  <h2>Overview</h2>
  <p>A medical chatbot is an advanced technological tool that leverages AI and natural language processing to simulate human-like interactions in the healthcare field. It acts as a virtual assistant, offering support, information, and assistance to users. The chatbot aims to enhance healthcare services by providing reliable and compassionate support.</p>

  <h2>Purpose</h2>
  <p>The purpose of a medical chatbot is to provide accessible and efficient healthcare support to individuals. Here are the key purposes of a medical chatbot:</p>
  <ol>
    <li>Information: Medical chatbots aim to offer accurate and up-to-date information about medical conditions, medications, and preventive measures. They serve as a reliable source of knowledge, providing users with educational resources and addressing common health-related questions.</li>
    <li>Symptom Evaluation and Triage: Chatbots can assist users in evaluating their symptoms and provide initial assessments of potential causes based on medical knowledge and algorithms. They can offer general advice or recommend seeking further medical attention if necessary.</li>
  </ol>

  <h2>Project Setup</h2>
  <p>To set up the project, follow these steps:</p>
  <ol>
    <li>Clone the Repository:</li>
    <pre>git clone https://github.com/yousefehab22/medical-chatbot.git</pre>
    <li>Create and Activate Virtual Environment:</li>
    <pre>conda create -n [venv_name] python=3.9</pre>
    <pre>conda activate [venv_name]</pre>
    <li>Install Required Libraries:</li>
    <pre>pip install -r requirements.txt</pre>
  </ol>

  <h2>Training</h2>
  <p>To train the model, run the following command:</p>
  <pre>python train_model.py</pre>

  <h2>Testing</h2>
  <p>To test the model, run the following command:</p>
  <pre>python test_model.py</pre>
  <p>Note: Please ensure that you check the paths mentioned in the code before running the project.</p>

  <h2>Directory Structure</h2>
  <ul>
    <li>model_path: Contains the trained model files.</li>
    <li>pickle_files_path: Contains the pickle files used for data preprocessing or other purposes.</li>
    <li>Dataset_path: Contains the dataset used for training the model.</li>
  </ul>
  <p>Please make sure to update the paths and commands according to your specific project setup.</p>
</body>
</html>
