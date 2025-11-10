/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunSubtaskResponse } from './RunSubtaskResponse';
export type RunWithSubtasksResponse = {
    created_at: string;
    id: number;
    is_hidden: boolean;
    run_name: string;
    subtasks: Array<RunSubtaskResponse>;
    yaml_path: string;
};
