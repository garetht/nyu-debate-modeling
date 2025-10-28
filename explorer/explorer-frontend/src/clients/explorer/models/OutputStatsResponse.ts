/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { DailyDebateStatsResponse } from './DailyDebateStatsResponse';
import type { DebateStatsSummary } from './DebateStatsSummary';
/**
 * Statistics summary for a configuration's transcripts.
 */
export type OutputStatsResponse = {
    configuration: string;
    errors: Array<string>;
    json_file_count: number;
    overall_stats: DebateStatsSummary;
    per_day: Array<DailyDebateStatsResponse>;
    transcripts_directory: string;
};

