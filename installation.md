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

11. # Next you can use type: LoadBalancer if you want to create a alb in the cloud
    - In AWS you will need domain, route53, ALB, Target groups, EKS with the services running 




