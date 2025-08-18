1. # Install Java
JAVA_HOME = C:\Program Files\Java\jdk-21\

PATH =<EXITSTIN GPATH> +%java_home
systems var	JAVA_HOME = C:\Program Files\Java\jdk-21\
add to  path	%JAVA_HOME%\bin
2. # Python
3. scikit-learn
pip install scikit_learn

3. # intsall matplotlib
pip install matplotlib

3. # Juypter lab
juypter lab to run the lab


4. # Install docker
- Install docker
- Create a repository in docker hub dockerak100/my_house_price. Login to docker hub and create a repo https://hub.docker.com/repository/docker/dockerak100/my_house_price/general
```bash
$ docker --version
$ docker images -a
$ docker ps -a
$ docker build . -t <<name-of-docker-hub/<<mae-of-repo>>
$ docker run -p 6000:5000 <<name-of-docker-hub/<<name-of-repo>>:latest # The contain port is 5000, the host port is 6000
$ docker push <<name-of-docker-hub/<<name-of-repo>>:latest
$ docker pull <<name-of-docker-hub/<<name-of-repo>>:latest # 
```

5. # Install kubernetes with kind
6. # Install kubectl
```bash
$ kubectl config get-contexts
$ kubectl config use-context docker-desktop
$ kubectl  get nodes
$ kubectl config get-contexts
```
7. # Create a pod mlapp-pod in namespace mlops-visual. Folder 02_building_ml_app/manifests/mlapp.yaml - creates a test pods
```bash
$ cd 02_building_ml_app/manifests/
$ kubectl create ns mlops-visual
$ kubectl apply -f .
$ kubectl get pods
$ kubectl get pods -n mlops_visual
$ kubectl delete -f mlapp.yaml
```
8. # Create a Deployment, 3 replicas to enable scaling
```bash
$ kubectl apply -f deployment.yaml
$ kubectl get deploy -n mlapp-visual
$ kubectl get pods -n mlapp-visual
$ kubectl get all -n mlapp-visual
```

9. # Create a service(distributes traffic across the nodes) and expose the 7000 port.
```bash
$ kubectl apply -f service.yaml
$ k port-forward -n mlops-visual svc/mlapp-service 7000:5000
```
10. # Check in postman as POST
    - http://127.0.0.1:6000/predict

11. # Next you can use type: LoadBalancer if you want to create a alb in the cloud- additional not  included
    - In AWS you will need domain, route53, ALB, Target groups, EKS with the services running.
12. # You need an S3 bucket to store data and model in s3 and it should be automated vi aws cli. 
    - The link to install and configure https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
    - Then create an IAM user with s3 Full access
```bash
    $ aws --version
    $ aws configure
    $ aws s3 ls
    $ aws s3 cp data/HousePrices.csv s3://visualmlops-ai-artifacts/data/
    $ aws s3 cp model/house_price_model.pkl s3://visualmlops-ai-artifacts/model/
```
13. # Download jar file from jenkins. 
```bash
   $ java -jar ~/jenkins/jenkins.war
```
14. # Go to browser type 
    - type localhost:8080 in browser
    - Install plugins
    - Create username/password
15. # Build the MLAPP
    - Creat a jenkins job
    - This jenkins job to pull the source code for mlapp from github 
      clone repo
    - Creates jobs for 3 stages
        python ./model.py
    - Build docker images
        docker build . -t <<name-of-docker-hub/<<name-of-repo>>:latest
    - Publish the image
        docker push <<name-of-docker-hub/<<name-of-repo>>:latest
16. # Deploy to k8s via new deploy jenkins job with a trigger 
    - kubectl delete -f manifests/mlapp-deployment.yaml
    - kubectl apply -f manifests/

17. # Intsall boto3
 ```bash
$ pip install boto3
```
18. # In model.py we upload the model to s3 path model/house_price_model.pkl
19. # In model we download form s3 and to local path model/house_price_model.pkl
    - make sure you do this befor you run create the falsk app
20. # Install dvc for version control
    - Follow these instructions to version control your data https://dvc.org/doc/start?tab=Mac-Linux
    - example history
```bash
$ git status
$ dvc init 
$ git commit -m "Initialize DVC"
$ dvc get https://github.com/iterative/dataset-registry           get-started/data.xml -o data/data.xml
$ dvc add data/data.xml
$ git add data/data.xml.dvc data/.gitignore
$ git commit -m "Add raw data"
$ mkdir /tmp/dvcstore
$ dvc remote add -d storage s3://visualmlops-ai-artifacts/dvcstore # nothing is added here. Empy folder is pushed to s3 
$ dvc push
$ aws s3 ls s3://visualmlops-ai-artifacts/dvcstore
```










