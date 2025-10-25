/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunDetailResponse } from '../models/RunDetailResponse';
import type { RunSubtaskResponse } from '../models/RunSubtaskResponse';
import type { RunWithSubtasksResponse } from '../models/RunWithSubtasksResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * Health Check
     * Simple health endpoint to confirm the API is responsive.
     * @returns string Successful Response
     * @throws ApiError
     */
    public static healthCheckHealthGet(): CancelablePromise<Record<string, string>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/health',
        });
    }
    /**
     * List Runs
     * Return all recorded runs along with their subtasks ordered by recency.
     * @returns RunWithSubtasksResponse Successful Response
     * @throws ApiError
     */
    public static listRunsRunsGet(): CancelablePromise<Array<RunWithSubtasksResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/runs',
        });
    }
    /**
     * Get Run
     * Return a single run and its associated subtasks.
     * @param runId
     * @returns RunDetailResponse Successful Response
     * @throws ApiError
     */
    public static getRunRunsRunIdGet(
        runId: number,
    ): CancelablePromise<RunDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/runs/{run_id}',
            path: {
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Run Subtasks
     * Return subtasks for a specific run.
     * @param runId
     * @returns RunSubtaskResponse Successful Response
     * @throws ApiError
     */
    public static listRunSubtasksRunsRunIdSubtasksGet(
        runId: number,
    ): CancelablePromise<Array<RunSubtaskResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/runs/{run_id}/subtasks',
            path: {
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Subtasks
     * Return all subtasks, optionally filtered by run identifier.
     * @param runId
     * @returns RunSubtaskResponse Successful Response
     * @throws ApiError
     */
    public static listSubtasksSubtasksGet(
        runId?: (number | null),
    ): CancelablePromise<Array<RunSubtaskResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/subtasks',
            query: {
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
