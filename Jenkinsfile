pipeline {
    agent {
        docker {
            image 'python:3.10-slim'
            args '-v $HOME/.cache/pip:/root/.cache/pip'
        }
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Diagnostics') {
            steps {
                // print helpful info when Docker or Python is missing
                sh '''
                echo "=== PATH / Tools ==="
                which docker || true
                docker --version || true
                which python || true
                python --version || true
                which pip || true
                pip --version || true
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                # prefer python -m pip, fall back to pip
                python -m pip install -r requirements.txt || python3 -m pip install -r requirements.txt || pip install -r requirements.txt
                '''
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