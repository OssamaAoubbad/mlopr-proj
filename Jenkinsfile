pipeline {
    agent any

    parameters {
        string(name: 'BRANCH', defaultValue: 'dev', description: 'Branch to build/deploy (e.g. uat, dev)')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                // ensure requested branch is checked out when running from a simple job
                sh "git fetch --all || true"
                sh "git checkout ${params.BRANCH} || true"
            }
        }

        stage('Prepare') {
            steps {
                script {
                    def dockerAvailable = sh(script: 'docker info >/dev/null 2>&1 && echo true || echo false', returnStdout: true).trim()
                    env.USE_DOCKER = dockerAvailable
                    echo "USE_DOCKER=${env.USE_DOCKER}"
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    if (env.USE_DOCKER == 'true') {
                        docker.image('python:3.10-slim').inside('-v $HOME/.cache/pip:/root/.cache/pip') {
                            sh 'python -m pip install -r requirements.txt'
                        }
                    } else {
                        sh '''
                            # Ensure pip exists (user install) and create a venv in workspace
                            python3 -m pip --version || (
                                (curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py || wget -q -O get-pip.py https://bootstrap.pypa.io/get-pip.py) && \
                                python3 get-pip.py --user
                            )
                            export PATH="$HOME/.local/bin:$PATH"

                            # Create a venv in the workspace (fallback to virtualenv if venv missing)
                            python3 -m venv .venv || (
                                python3 -m pip install --user virtualenv && \
                                python3 -m virtualenv .venv
                            )

                            # Activate and install requirements into venv
                            . .venv/bin/activate
                            python -m pip install --upgrade pip
                            python -m pip install -r requirements.txt
                        '''
                    }
                }
            }
        }

        stage('Train') {
            steps {
                script {
                    if (env.USE_DOCKER == 'true') {
                        docker.image('python:3.10-slim').inside('-v $HOME/.cache/pip:/root/.cache/pip') {
                            sh 'python -m madewithml.train --experiment-name mlops-project --dataset-loc datasets/dataset.csv'
                        }
                    } else {
                        sh '. .venv/bin/activate && python -m madewithml.train --experiment-name mlops-project --dataset-loc datasets/dataset.csv'
                    }
                }
            }
        }

        stage('Evaluate') {
            steps {
                script {
                    if (env.USE_DOCKER == 'true') {
                        docker.image('python:3.10-slim').inside('-v $HOME/.cache/pip:/root/.cache/pip') {
                            sh 'python -m madewithml.evaluate --run-id $(cat results.json | python -c "import sys,json; print(json.load(sys.stdin)[\"run_id\"])")'
                        }
                    } else {
                        sh '. .venv/bin/activate && python -m madewithml.evaluate --run-id $(cat results.json | python -c "import sys,json; print(json.load(sys.stdin)[\"run_id\"])")'
                    }
                }
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