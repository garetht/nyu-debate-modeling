/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Summary information about a single output configuration.
 */
export type OutputSummaryResponse = {
    config_type: OutputSummaryResponse.config_type;
    configuration: string;
    debater_key: string;
    debater_training: string;
    directory_size_bytes: number;
    judge_key: string;
    judge_training: string;
    latest_transcript: (string | null);
    task_label: string;
    transcript_count: number;
    transcripts_by_day: Record<string, number>;
    transcripts_directory: string;
};
export namespace OutputSummaryResponse {
    export enum config_type {
        EVAL = 'eval',
        DATA_GENERATION = 'data-generation',
    }
}

