/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { OutputGroupResponse } from './OutputGroupResponse';
import type { OutputSummaryResponse } from './OutputSummaryResponse';
/**
 * Response payload for listing available outputs.
 */
export type OutputsListResponse = {
    entries: Array<OutputSummaryResponse>;
    group_mode: string;
    groups: Array<OutputGroupResponse>;
    outputs_directory: string;
};

