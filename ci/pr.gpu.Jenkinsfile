pipeline {
    agent {
        docker {
            image 'nvcr.io/nvstaging/merlin/merlin-ci-runner-wrapper'
            label 'merlin_gpu_gcp || merlin_gpu'
            registryCredentialsId 'jawe-nvcr-io'
            registryUrl 'https://nvcr.io'
            args "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"
        }
    }

    options {
      buildDiscarder(logRotator(numToKeepStr: '10'))
      ansiColor('xterm')
      disableConcurrentBuilds(abortPrevious: true)
    }

    stages {
        stage("test-gpu") {
            options {
                timeout(time: 60, unit: 'MINUTES', activity: true)
            }
            steps {
                sh """#!/bin/bash
set -e
printenv

rm -rf \$HOME/.cudf/
export TF_MEMORY_ALLOCATION="0.1"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
export MKL_SERVICE_FORCE_INTEL=1
export NR_USER=true

pip install testbook
WANDB_API_KEY=a0bb504f852b51d3dbfa6f3f2d14184600db210e tox -re py38-gpu
                """
            }
        }
    }
}