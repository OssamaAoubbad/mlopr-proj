pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train') {
            steps {
                sh 'python -m madewithml.train --experiment-name mlops-project --dataset-loc datasets/dataset.csv'
            }
        }

        stage('Evaluate') {
            steps {
                sh 'python -m madewithml.evaluate --run-id $(cat results.json | python -c "import sys,json; print(json.load(sys.stdin)[\"run_id\"])")'
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}