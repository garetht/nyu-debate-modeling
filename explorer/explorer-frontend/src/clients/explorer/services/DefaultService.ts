/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { OutputDetailResponse } from '../models/OutputDetailResponse';
import type { OutputsListResponse } from '../models/OutputsListResponse';
import type { OutputStatsResponse } from '../models/OutputStatsResponse';
import type { RunDetailResponse } from '../models/RunDetailResponse';
import type { RunProcessResponse } from '../models/RunProcessResponse';
import type { RunSubtaskResponse } from '../models/RunSubtaskResponse';
import type { RunWithSubtasksResponse } from '../models/RunWithSubtasksResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * List Outputs
     * List available output configurations with optional grouping.
     * @param groupMode
     * @returns OutputsListResponse Successful Response
     * @throws ApiError
     */
    public static listOutputsApiOutputsGet(
        groupMode?: (string | null),
    ): CancelablePromise<OutputsListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/outputs',
            query: {
                'group_mode': groupMode,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Output Configuration
     * Return detailed information and transcripts for a configuration.
     * @param configuration
     * @param page
     * @param pageSize
     * @returns OutputDetailResponse Successful Response
     * @throws ApiError
     */
    public static getOutputConfigurationApiOutputsConfigurationGet(
        configuration: string,
        page: number = 1,
        pageSize: number = 100,
    ): CancelablePromise<OutputDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/outputs/{configuration}',
            path: {
                'configuration': configuration,
            },
            query: {
                'page': page,
                'page_size': pageSize,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Output Configuration Stats
     * Return debate statistics for an evaluation configuration.
     * @param configuration
     * @returns OutputStatsResponse Successful Response
     * @throws ApiError
     */
    public static getOutputConfigurationStatsApiOutputsConfigurationStatsGet(
        configuration: string,
    ): CancelablePromise<OutputStatsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/outputs/{configuration}/stats',
            path: {
                'configuration': configuration,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
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
     * List Run Processes
     * Return remote process metadata for each subtask associated with a run.
     * @param runId
     * @returns RunProcessResponse Successful Response
     * @throws ApiError
     */
    public static listRunProcessesRunsRunIdProcessesGet(
        runId: number,
    ): CancelablePromise<Array<RunProcessResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/runs/{run_id}/processes',
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
