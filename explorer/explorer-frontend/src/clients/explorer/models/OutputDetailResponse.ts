/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TranscriptFileResponse } from './TranscriptFileResponse';
/**
 * Detailed view of a specific output configuration and its transcripts.
 */
export type OutputDetailResponse = {
    config_type: 'eval' | 'data-generation';
    configuration: string;
    debater_key: string;
    debater_training: string;
    directory_size_bytes: number;
    judge_key: string;
    judge_training: string;
    latest_transcript: (string | null);
    page: number;
    page_size: number;
    task_label: string;
    total_pages: number;
    total_transcripts: number;
    transcript_count: number;
    transcripts: Array<TranscriptFileResponse>;
    transcripts_by_day: Record<string, number>;
    transcripts_directory: string;
};

