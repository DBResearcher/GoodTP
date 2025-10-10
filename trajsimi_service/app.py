# -*- encoding: utf-8 -*-
# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Tuple
import uvicorn
import traceback

from similarity_utils import compute_batch_simi_matrices, logger, DATASET_MAP

# 请求体的格式定义
class TrajSimiRequest(BaseModel):
    method: str
    # 批量轨迹数据，每个batch是一个轨迹列表
    batch_trajectories: List[List[List[List[float]]]]
    # 对应的轨迹ID列表
    batch_traj_ids: List[List[int]]
    # 数据集名称
    dataset: str
    use_cache: bool

app = FastAPI()

@app.post("/compute_similarity")
def compute_similarity(data: TrajSimiRequest):
    """
    接收批量轨迹列表和要使用的相似度方法，计算相似度矩阵并返回
    """
    try:
        # 验证数据集是否支持
        if data.dataset not in DATASET_MAP:
            raise ValueError(f"Unsupported dataset: {data.dataset}. Supported datasets: {set(DATASET_MAP.keys())}")
        
        # 验证批量数据长度是否一致
        if len(data.batch_trajectories) != len(data.batch_traj_ids):
            raise ValueError("batch_trajectories and batch_traj_ids must have the same length")
        
        # 验证每个batch内的轨迹数量与ID数量是否一致
        for i, (trajectories, traj_ids) in enumerate(zip(data.batch_trajectories, data.batch_traj_ids)):
            if len(trajectories) != len(traj_ids):
                raise ValueError(f"Batch {i}: trajectories count ({len(trajectories)}) != traj_ids count ({len(traj_ids)})")
        
        # 批量计算相似度矩阵
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
        # 如果 method 不支持，抛出 400
        error_details = traceback.format_exc()
        logger.error("Value Error:\n", error_details)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 其他未知错误
        error_details = traceback.format_exc()
        logger.error("Internal Server Error:\n", error_details)
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")
    
    return {
        "method": data.method,
        "dataset": data.dataset,
        "batch_matrix": batch_matrices
    }



if __name__ == "__main__":
    # 开发环境直接用 uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
