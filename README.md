# When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions
Code for www-24 paper: [When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions](https://arxiv.org/pdf/2305.12650).

## Abatract
Federated recommendation systems usually trains a global model on the server without direct access to users’ private data on their own devices. However, this separation of the recommendation model and users’ private data poses a challenge in providing quality service, particularly when it comes to new items, namely cold-start recommendations in federated settings. This paper introduces a novel method called Item-aligned Federated Aggregation (IFedRec) to address this challenge. It is the first research work in federated recommendation to specifically study the cold-start scenario. The proposed method learns two sets of item representations by leveraging item attributes and interaction records simultaneously. Additionally, an item representation alignment mechanism is designed to align two item representations and learn the meta attribute network at the server within a federated learning framework. Experiments on four benchmark datasets demonstrate IFedRec’s superior performance for cold-start scenarios. Furthermore, we also verify IFedRec owns good robustness when the system faces limited client participation and noise injection, which brings promising practical application potential in privacy-protection enhanced federated recommendation systems. 

![](https://github.com/Zhangcx19/IFedRec/blob/main/comparison.png)
**Figure:**
Three cold-start recommendation systems comparison. The centralized method (a) saves raw item attributes on the server but exposes private user interaction records. Traditional FedRecSys (b) secures the interaction records but exposes the item attributes to the clients. Our IFedRec can protect these two types of security-sensitive information.

## Directory IFedNCF (IPFedRec).
The implentation code of integrating the state-of-the-art federated recommendation models FedNCF (PFedRec) into our proposed framework to achieve the cold-start recommendation.

## Directory data/CiteULike.
The benchmark cold-start recommendation evaluation dataset CiteULike, which is evaluated in our expereiments.

## Preparations before running the code
mkdir log

mkdir sh_result

## Running the code
python train.py

## Citation
If you find this project helpful, please consider to cite the following paper:

```
@inproceedings{zhang2024federated,
  title={When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions},
  author={Zhang, Chunxu and Long, Guodong and Zhou, Tianyi and Zhang, Zijian and Yan, Peng and Yang, Bo},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={3632--3642},
  year={2024}
}
```
