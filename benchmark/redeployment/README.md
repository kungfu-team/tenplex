# Redeployment
_Fig. 10. Redeployment time of DL job_

We evaluate how long Scalai takes to redeploy DL jobs with different model sizes onto a new set of GPU resources. As a baseline, we compare against Scalai-Central, which follows the approach of PyTorch Elastic  or DeepSpeed: it holds all DL job state at a single central worker. In this experiment, we therefore specifically explore the benefit of Scalai’s distributed state management.