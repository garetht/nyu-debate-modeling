/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunSubtaskResponse } from './RunSubtaskResponse';
import type { RunTaskResponse } from './RunTaskResponse';
export type RunDetailResponse = {
    run: RunTaskResponse;
    subtasks: Array<RunSubtaskResponse>;
};

