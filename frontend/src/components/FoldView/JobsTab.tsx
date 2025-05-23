import React from "react";
import { Invokation } from "../../types/types";
import { TabContainer, TableSection, SectionCard, ResponsiveTable } from "../../util/tabComponents";

interface JobsTabProps {
    jobs: Invokation[] | null;
}

const JobsTab: React.FC<JobsTabProps> = ({ jobs }) => {
    const formatStartTime = (jobstarttime: string | null) => {
        if (!jobstarttime) return "Not Started / Unknown";

        try {
            // Parse the UTC time string into a Date object
            const date = new Date(jobstarttime);

            if (isNaN(date.getTime())) {
                console.warn(`Invalid date value ${jobstarttime}`);
                return "Invalid date";
            }
            return new Intl.DateTimeFormat('en-US', {
                timeStyle: "short",
                dateStyle: "short",
                timeZone: "America/Los_Angeles"
            }).format(date);
        } catch (error) {
            console.error(`Error formatting date ${jobstarttime}:`, error);
            return "Error";
        }
    };

    const formatRunTime = (jobRunTime: number | null) => {
        return jobRunTime
            ? `${Math.floor(jobRunTime / (60 * 60))} hr ${Math.floor(jobRunTime / 60) % 60
            } min ${Math.floor(jobRunTime) % 60} sec`
            : "NA";
    };

    if (!jobs) return null;

    return (
        <TabContainer>
            <TableSection title="Invokations">
                <ResponsiveTable>
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th className="uk-text-nowrap">State</th>
                            <th className="uk-text-nowrap">Logs</th>
                            <th className="uk-text-nowrap">Start time</th>
                            <th className="uk-text-nowrap">Runtime</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[...jobs].map((job: Invokation) => (
                            <tr key={`${job.job_id}_${job.id}`}>
                                <td className="uk-text-nowrap" uk-tooltip={job.type}>
                                    {job.type}
                                </td>
                                <td className="uk-text-nowrap" uk-tooltip={job.state}>
                                    {job.state}
                                </td>
                                <td className="uk-text-nowrap">
                                    <a href={`#logs_${job.id?.toString()}`}>View</a>
                                </td>
                                <td
                                    className="uk-text-nowrap"
                                    uk-tooltip={formatStartTime(job.starttime)}
                                >
                                    {formatStartTime(job.starttime)}
                                </td>
                                <td
                                    className="uk-text-nowrap"
                                    uk-tooltip={formatRunTime(job.timedelta_sec)}
                                >
                                    {formatRunTime(job.timedelta_sec)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </ResponsiveTable>
            </TableSection>

            {/* Job Logs */}
            {[...jobs].map((job: Invokation) => (
                <SectionCard
                    key={job.id || "jobid should not be null"}
                    style={{ marginBottom: '20px' }}
                >
                    <div id={`logs_${job.id?.toString()}`}>
                        <h3 style={{ marginBottom: '15px', overflowWrap: 'anywhere' }}>{job.type} Logs</h3>
                        <div style={{
                            backgroundColor: '#f8f9fa',
                            padding: '15px',
                            borderRadius: '4px',
                            marginBottom: '10px',
                            overflowX: 'auto'
                        }}>
                            <strong>Command:</strong> {job.command}
                        </div>
                        <pre style={{
                            backgroundColor: '#f8f9fa',
                            padding: '15px',
                            borderRadius: '4px',
                            overflowX: 'auto',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word'
                        }}>
                            {job.log}
                        </pre>
                    </div>
                </SectionCard>
            ))}
        </TabContainer>
    );
};

export default JobsTab;
