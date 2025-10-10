# -*- encoding: utf-8 -*-
# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Tuple
import uvicorn
import traceback

from similarity_utils import compute_batch_simi_matrices, logger, DATASET_MAP

# request body format definition
class TrajSimiRequest(BaseModel):
    method: str
    # batch of trajectories, each batch is a list of trajectories
    batch_trajectories: List[List[List[List[float]]]]
    # corresponding trajectory IDs
    batch_traj_ids: List[List[int]]
    # dataset name
    dataset: str
    use_cache: bool

app = FastAPI()

@app.post("/compute_similarity")
def compute_similarity(data: TrajSimiRequest):
    """
    receive batch of trajectories and the similarity method to use, compute the similarity matrix and return
    """
    try:
        # verify if the dataset is supported
        if data.dataset not in DATASET_MAP:
            raise ValueError(f"Unsupported dataset: {data.dataset}. Supported datasets: {set(DATASET_MAP.keys())}")
        
        # verify if the length of the batch data is consistent
        if len(data.batch_trajectories) != len(data.batch_traj_ids):
            raise ValueError("batch_trajectories and batch_traj_ids must have the same length")
        
        # verify if the number of trajectories in each batch is consistent with the number of IDs
        for i, (trajectories, traj_ids) in enumerate(zip(data.batch_trajectories, data.batch_traj_ids)):
            if len(trajectories) != len(traj_ids):
                raise ValueError(f"Batch {i}: trajectories count ({len(trajectories)}) != traj_ids count ({len(traj_ids)})")
        
        # batch compute similarity matrices
        if data.use_cache:
            processes = 20
        else:
            processes = 1
        batch_matrices = compute_batch_simi_matrices(
            data.method, 
            data.batch_trajectories, 
            data.batch_traj_ids, 
            data.dataset,
            data.use_cache,
            processes = processes
        )
        
    except ValueError as e:
        # if the method is not supported, raise 400
        error_details = traceback.format_exc()
        logger.error("Value Error:\n", error_details)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # other unknown errors
        error_details = traceback.format_exc()
        logger.error("Internal Server Error:\n", error_details)
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
    
    return {
        "method": data.method,
        "dataset": data.dataset,
        "batch_matrix": batch_matrices
    }



if __name__ == "__main__":
    # use uvicorn in development environment
    uvicorn.run(app, host="0.0.0.0", port=8800)
