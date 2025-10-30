import React, { useMemo, useState } from "react";
import { FaCheckCircle } from "react-icons/fa";
import { Spin, Table, Tooltip } from "antd";
import { ColumnsType } from 'antd/es/table';
import { describeFoldState, getFoldAffinityPrediction, updateFold } from "../api/foldApi";
import { AffinityPrediction, Fold } from "../types/types";
import { BoltzYamlHelper } from "./boltzYamlHelper";
import { EditableTagList } from "./editableTagList";
import { notify } from "../services/NotificationService";
import { Link } from "react-router-dom";

interface FoldTagsProps {
    fold: Fold;
    onTagsChange?: () => void;
    userType?: string | null;
    editable?: boolean;
}

function FoldTags({ fold, onTagsChange, userType, editable = false }: FoldTagsProps) {
    const [isUpdating, setIsUpdating] = useState(false);

    const addTag = async (tagToAdd: string) => {
        if (!fold.tags.includes(tagToAdd) && fold.id !== null) {
            setIsUpdating(true);
            try {
                const newTags = [...fold.tags, tagToAdd];
                await updateFold(fold.id, { tags: newTags });
                onTagsChange?.();
                notify.success("Tag added successfully");
            } catch (error) {
                notify.error(`Failed to add tag: ${error}`);
            } finally {
                setIsUpdating(false);
            }
        }
    };

    const deleteTag = async (tagToDelete: string) => {
        if (fold.id !== null) {
            setIsUpdating(true);
            try {
                const newTags = fold.tags.filter(tag => tag !== tagToDelete);
                await updateFold(fold.id, { tags: newTags });
                onTagsChange?.();
                notify.success("Tag removed successfully");
            } catch (error) {
                notify.error(`Failed to remove tag: ${error}`);
            } finally {
                setIsUpdating(false);
            }
        }
    };

    const handleTagClick = (tag: string) => {
        window.open(`/tag/${tag}`, "_self");
    };

    const isViewOnly = !editable || userType === "viewer";

    return (
        <div style={{ opacity: isUpdating ? 0.6 : 1, pointerEvents: isUpdating ? 'none' : 'auto' }}>
            <EditableTagList
                tags={fold.tags}
                addTag={addTag}
                deleteTag={deleteTag}
                handleTagClick={handleTagClick}
                viewOnly={isViewOnly}
            />
        </div>
    );
}

interface AffinityResult {
    binder_id: string;
    affinity: AffinityPrediction | null;
}

type AffinityState = 'loading' | 'loaded' | 'failed' | 'no-affinity';

interface FoldAffinityProps {
    foldId: number | null;
    foldYamlHelper: BoltzYamlHelper | null;
}

function FoldAffinityCell({ foldId, foldYamlHelper }: FoldAffinityProps) {
    const [affinityResult, setAffinityResult] = useState<AffinityResult | null>(null);
    const [affinityState, setAffinityState] = useState<AffinityState>('loading');

    useMemo(() => {
        if (foldId === null || !foldYamlHelper) {
            setAffinityState('no-affinity');
            return null;
        }

        const properties = foldYamlHelper.getProperties();
        const affinityProperty = properties?.find(p => 'affinity' in p);
        const affinityBinderId = affinityProperty?.affinity?.binder;
        if (!affinityBinderId) {
            setAffinityState('no-affinity');
            return null;
        }

        setAffinityState('loading');

        getFoldAffinityPrediction(foldId).then(
            (predictedAffinity: AffinityPrediction) => {
                setAffinityResult({
                    binder_id: affinityBinderId,
                    affinity: predictedAffinity,
                });
                setAffinityState('loaded');
            },
            (e: any) => {
                console.log(e);
                setAffinityResult({
                    binder_id: affinityBinderId,
                    affinity: null,
                });
                setAffinityState('failed');
            }
        );
    }, [foldId, foldYamlHelper]);

    const binderId = affinityResult?.binder_id || '';

    if (affinityState === 'no-affinity') {
        return { target: binderId, prediction: null };
    }

    if (affinityState === 'loading') {
        return { target: binderId, prediction: <Spin size="small" /> };
    }

    if (!affinityResult || affinityState === 'failed' || affinityResult.affinity === null) {
        return { target: binderId, prediction: <i>-</i> };
    }

    return {
        target: binderId,
        prediction: `${Math.pow(10, affinityResult.affinity.affinity_pred_value).toPrecision(2)} μM`
    };
}

interface FoldTableOptions {
    editable?: boolean;
    userType?: string | null;
    onTagsChange?: () => void;
}

export function makeFoldTableAntd(folds: Fold[], options: FoldTableOptions = {}) {
    const { editable = false, userType = null, onTagsChange } = options;

    const columns: ColumnsType<Fold> = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            width: 200,
            ellipsis: {
                showTitle: false,
            },
            render: (name: string, record: Fold) => (
                <Tooltip title={name}>
                    <Link
                        to={"/fold/" + record.id}
                        style={{
                            fontSize: '16px',
                            fontWeight: 600,
                            textDecoration: 'none',
                            transition: 'all 0.2s',
                            color: '#1890ff',
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.textDecoration = 'underline';
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.textDecoration = 'none';
                        }}
                    >
                        {name}
                    </Link>
                </Tooltip>
            ),
        },
        {
            title: 'Length',
            key: 'length',
            width: 70,
            render: (_, record: Fold) => (
                record.sequence?.length ||
                record.yaml_helper?.getProteinSequences().reduce((accumulator, cs) => accumulator + cs[1].length, 0)
            ),
        },
        {
            title: 'State',
            key: 'state',
            width: 70,
            ellipsis: {
                showTitle: false,
            },
            render: (_, record: Fold) => {
                const state = describeFoldState(record);
                return (
                    <Tooltip title={state}>
                        <span>{state}</span>
                    </Tooltip>
                );
            },
        },
        {
            title: 'Owner',
            dataIndex: 'owner',
            key: 'owner',
            width: 120,
            ellipsis: {
                showTitle: false,
            },
            render: (owner: string) => (
                <Tooltip title={owner}>
                    <span>{owner}</span>
                </Tooltip>
            ),
        },
        {
            title: 'Date Created',
            dataIndex: 'create_date',
            key: 'create_date',
            width: 100,
            ellipsis: true,
            render: (date: string) => {
                try {
                    const dateObj = new Date(date);
                    if (isNaN(dateObj.getTime())) {
                        console.warn(`Invalid date value: ${date}`);
                        return "Invalid date";
                    }
                    return new Intl.DateTimeFormat('en-US', {
                        timeStyle: "short",
                        dateStyle: "short",
                        timeZone: "America/Los_Angeles"
                    }).format(dateObj);
                } catch (error) {
                    console.error(`Error formatting date:`, error);
                    return "Error";
                }
            },
        },
        {
            title: 'Public',
            dataIndex: 'public',
            key: 'public',
            width: 50,
            render: (isPublic: boolean) => (
                isPublic ? (
                    <Tooltip title="This fold is visible to the public.">
                        <FaCheckCircle />
                    </Tooltip>
                ) : null
            ),
        },
        {
            title: 'Tags',
            key: 'tags',
            width: 100,
            render: (_, record: Fold) => (
                <div style={{ overflowX: 'auto' }}>
                    <FoldTags
                        fold={record}
                        onTagsChange={onTagsChange}
                        userType={userType}
                        editable={editable}
                    />
                </div>
            ),
        },
        {
            title: <div>Affinity<br />Target</div>,
            key: 'affinity_target',
            width: 50,
            render: (_, record: Fold) => {
                const AffinityComponent = () => {
                    const result = FoldAffinityCell({
                        foldId: record.id,
                        foldYamlHelper: record.yaml_helper
                    });
                    return <>{result.target}</>;
                };
                return <AffinityComponent />;
            },
        },
        {
            title: <div>Affinity<br />Prediction</div>,
            key: 'affinity_prediction',
            width: 100,
            render: (_, record: Fold) => {
                const AffinityComponent = () => {
                    const result = FoldAffinityCell({
                        foldId: record.id,
                        foldYamlHelper: record.yaml_helper
                    });
                    return <>{result.prediction}</>;
                };
                return <AffinityComponent />;
            },
        },
    ];

    return (
        <Table
            columns={columns}
            dataSource={folds}
            rowKey={(record) => record.name}
            pagination={false}
            size="small"
            scroll={{ x: 800 }}
            onRow={() => ({
                style: {
                    transition: 'background-color 0.2s',
                },
                onMouseEnter: (e: React.MouseEvent<HTMLTableRowElement>) => {
                    e.currentTarget.style.backgroundColor = '#fafafa';
                },
                onMouseLeave: (e: React.MouseEvent<HTMLTableRowElement>) => {
                    e.currentTarget.style.backgroundColor = '';
                },
            })}
        />
    );
}
