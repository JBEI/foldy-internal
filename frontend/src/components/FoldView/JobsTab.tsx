import React from "react";
import { Invokation } from "../../types/types";
import { TabContainer, TableSection, SectionCard } from "../../util/tabComponents";
import { AntTable, defaultExpandableContent } from "../../util/AntTable";

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
                <AntTable<Invokation>
                    dataSource={jobs || []}
                    rowKey={(record) => `${record.job_id}_${record.id}`}
                    expandableContent={defaultExpandableContent}
                    columns={[
                        {
                            key: 'type',
                            title: 'Type',
                            dataIndex: 'type',
                            ellipsis: true,
                            render: (type) => <span title={type}>{type}</span>,
                        },
                        {
                            key: 'state',
                            title: 'State',
                            dataIndex: 'state',
                            width: 100,
                            render: (state) => <span title={state}>{state}</span>,
                        },
                        {
                            key: 'logs',
                            title: 'Logs',
                            width: 80,
                            render: (_, job) => (
                                <a href={`#logs_${job.id?.toString()}`}>View</a>
                            ),
                        },
                        {
                            key: 'starttime',
                            title: 'Start time',
                            width: 120,
                            render: (_, job) => {
                                const formatted = formatStartTime(job.starttime);
                                return <span title={formatted}>{formatted}</span>;
                            },
                        },
                        {
                            key: 'runtime',
                            title: 'Runtime',
                            width: 120,
                            render: (_, job) => {
                                const formatted = formatRunTime(job.timedelta_sec);
                                return <span title={formatted}>{formatted}</span>;
                            },
                        },
                    ]}
                />
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
