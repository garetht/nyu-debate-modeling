/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunSubtaskConfigurationName } from './RunSubtaskConfigurationName';
export type RunSubtaskResponse = {
    base_task_configuration?: RunSubtaskConfigurationName;
    base_task_name: string;
    command: string;
    configuration: Record<string, any>;
    created_at: string;
    id: number;
    ip_address: string;
    log_path: string;
    logs_command: string;
    resolved_task_name: string;
    run_task_id: number;
};
