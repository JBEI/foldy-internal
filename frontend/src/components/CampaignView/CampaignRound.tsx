import React, { useState, useEffect, useCallback } from 'react';
import {
    Card,
    Typography,
    Button,
    Modal,
    Spin,
} from 'antd';
import {
    ExperimentOutlined,
    DeleteOutlined,
    ExclamationCircleOutlined
} from '@ant-design/icons';
import type { Campaign, CampaignRound, Fold } from '../../types/types';
import { getFold } from '../../api/foldApi';
import { getCampaignRoundActivityData, deleteCampaignRound, updateCampaignRound } from '../../api/campaignApi';
import { getFile } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import SlateBuilder from '../shared/SlateBuilder';
import ZeroShotCampaignRoundView from './ZeroShotCampaignRoundView';
import FewShotCampaignRoundView from './FewShotCampaignRoundView';

const { Title, Text } = Typography;

interface CampaignRoundComponentProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    onRefresh: () => void;
    onRefreshRound?: () => void;
    onStartNewRound: () => void;
    onDeleteRound?: (roundId: number) => void;
}

const CampaignRoundComponent: React.FC<CampaignRoundComponentProps> = ({
    campaign,
    currentRound,
    onRefresh,
    onRefreshRound,
    onStartNewRound,
    onDeleteRound
}) => {
    const [loading, setLoading] = useState(true);
    const [fold, setFold] = useState<Fold | null>(null);
    const [naturalnessCsvData, setNaturalnessCsvData] = useState<string | null>(null);
    const [showSlateBuilder, setShowSlateBuilder] = useState(false);
    const [slateBuilderSeqIds, setSlateBuilderSeqIds] = useState<string[]>([]);
    const [activityData, setActivityData] = useState<Array<{ seq_id: string, activity: number }> | null>(null);

    // Load naturalness CSV data when needed
    const loadNaturalnessCsvData = useCallback(async (naturalness: any) => {
        if (!naturalness.output_fpath_computed) {
            notify.error('Naturalness run output file not found');
            return;
        }

        try {
            const fileBlob = await getFile(campaign.fold_id, naturalness.output_fpath_computed);
            const reader = new FileReader();
            reader.onload = (e) => {
                const fileString = e.target?.result as string;
                setNaturalnessCsvData(fileString);
            };
            reader.readAsText(fileBlob);
        } catch (error) {
            notify.error('Failed to load naturalness data');
            console.error('Error loading naturalness CSV:', error);
        }
    }, [campaign.fold_id]);

    // Load fold data
    const loadFoldData = useCallback(async () => {
        setLoading(true);
        try {
            const foldData = await getFold(campaign.fold_id);
            setFold(foldData);
        } catch (error) {
            notify.error('Failed to load fold data');
            console.error('Error loading fold:', error);
        } finally {
            setLoading(false);
        }
    }, [campaign.fold_id]);

    // Load activity data
    const loadActivityData = useCallback(async () => {
        if (!campaign?.id || !currentRound.result_activity_fpath) return;

        try {
            const response = await getCampaignRoundActivityData(campaign.id, currentRound.round_number);
            setActivityData(response.data);
        } catch (error: any) {
            console.error('Failed to load activity data:', error);
            notify.error('Failed to load activity data');
        }
    }, [campaign?.id, currentRound.result_activity_fpath, currentRound.round_number]);

    // Effects
    useEffect(() => {
        loadFoldData();
    }, [loadFoldData]);

    useEffect(() => {
        // Load CSV data if a naturalness run is selected
        if (currentRound.naturalness_run && currentRound.naturalness_run.output_fpath_computed) {
            loadNaturalnessCsvData(currentRound.naturalness_run);
        }
    }, [currentRound.naturalness_run_id, currentRound.naturalness_run, loadNaturalnessCsvData]);

    useEffect(() => {
        if (currentRound.result_activity_fpath && campaign?.id) {
            loadActivityData();
        }
    }, [currentRound.result_activity_fpath, campaign?.id, currentRound.round_number, loadActivityData]);

    const handleMeasurementChoice = async (hasMeasurements: boolean) => {
        const mode = hasMeasurements ? 'few-shot' : 'zero-shot';

        try {
            await updateCampaignRound(campaign.id!, currentRound.id, { mode });
            onRefresh();
        } catch (error) {
            notify.error('Failed to update campaign round mode');
            console.error('Error updating round mode:', error);
        }
    };

    const buildSlate = (seqIds: string[]) => {
        setSlateBuilderSeqIds(seqIds);
        setShowSlateBuilder(true);
    };

    const handleSlateConfirm = async (selectedSeqIds: string[]) => {
        try {
            const slateSeqIdsString = selectedSeqIds.join(',');
            await updateCampaignRound(campaign.id!, currentRound.id, { slate_seq_ids: slateSeqIdsString });
            onRefresh();
            notify.success(`Added ${selectedSeqIds.length} mutants to slate`);
        } catch (error) {
            notify.error('Failed to update slate');
            console.error('Error updating slate:', error);
        }
    };

    const handleDeleteRound = () => {
        Modal.confirm({
            title: 'Delete Round',
            icon: <ExclamationCircleOutlined />,
            content: (
                <div>
                    <p>Are you sure you want to delete <strong>Round {currentRound.round_number}</strong>?</p>
                    <p>This action cannot be undone and will permanently delete:</p>
                    <ul>
                        <li>All round data and settings</li>
                        <li>Mutant slate ({currentRound.slate_seq_ids ? currentRound.slate_seq_ids.split(',').length : 0} sequences)</li>
                        {currentRound.result_activity_fpath && <li>Activity results data</li>}
                        <li>Any associated naturalness run selections</li>
                    </ul>
                </div>
            ),
            okText: 'Delete Round',
            okType: 'danger',
            cancelText: 'Cancel',
            onOk: async () => {
                try {
                    await deleteCampaignRound(campaign.id!, currentRound.id);
                    notify.success(`Round ${currentRound.round_number} deleted successfully`);
                    if (onDeleteRound) {
                        onDeleteRound(currentRound.id);
                    }
                    onRefresh();
                } catch (error) {
                    notify.error('Failed to delete round');
                    console.error('Error deleting round:', error);
                }
            }
        });
    };

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '48px' }}>
                <Spin size="large" />
                <div style={{ marginTop: '16px' }}>
                    <Text type="secondary">Loading workflow...</Text>
                </div>
            </div>
        );
    }

    if (!fold) {
        return <div>Error: Could not load fold data</div>;
    }

    // Determine workflow step based on round and mode
    const renderWorkflowContent = () => {
        // First round or no mode set - show onboarding
        if (!currentRound.mode && currentRound.round_number === 1) {
            return (
                <Card>
                    <div style={{ textAlign: 'center', marginBottom: '24px' }}>
                        <ExperimentOutlined style={{ fontSize: '48px', color: '#1890ff', marginBottom: '16px' }} />
                        <Title level={3}>Welcome to Round {currentRound.round_number}</Title>
                        <Text type="secondary">
                            Let's set up your prediction workflow. First, we need to know about your experimental data.
                        </Text>
                    </div>

                    <div style={{ marginBottom: '24px' }}>
                        <Text strong>Do you have any measurements of mutant activity yet?</Text>
                        <br />
                        <Text type="secondary">This will help us choose the best prediction approach for your campaign.</Text>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '16px' }}>
                        <Button
                            type="primary"
                            size="large"
                            onClick={() => handleMeasurementChoice(true)}
                        >
                            Yes (I want to make few-shot predictions)
                        </Button>
                        <Button
                            size="large"
                            onClick={() => handleMeasurementChoice(false)}
                        >
                            No (I want to make zero-shot predictions)
                        </Button>
                    </div>
                </Card>
            );
        }

        // Round > 1 with no mode - default to few-shot
        if (!currentRound.mode && currentRound.round_number > 1) {
            handleMeasurementChoice(true);
            return null;
        }

        // Route to appropriate subcomponent based on mode
        if (currentRound.mode === 'zero-shot') {
            return (
                <ZeroShotCampaignRoundView
                    campaign={campaign}
                    currentRound={currentRound}
                    fold={fold}
                    naturalnessCsvData={naturalnessCsvData}
                    activityData={activityData}
                    onRefresh={onRefresh}
                    onRefreshRound={onRefreshRound}
                    buildSlate={buildSlate}
                />
            );
        } else if (currentRound.mode === 'few-shot') {
            return (
                <FewShotCampaignRoundView
                    campaign={campaign}
                    currentRound={currentRound}
                    fold={fold}
                    activityData={activityData}
                    onRefresh={onRefresh}
                    onRefreshRound={onRefreshRound}
                    buildSlate={buildSlate}
                />
            );
        }

        return <div>Unknown workflow mode</div>;
    };

    const hasActivityData = !!currentRound.result_activity_fpath;
    const isLastRound = !campaign.rounds || campaign.rounds.length === 0 ||
        currentRound.round_number === Math.max(...campaign.rounds.map(r => r.round_number));
    const nextRoundExists = campaign.rounds?.some(round => round.round_number === currentRound.round_number + 1) || false;

    return (
        <div>
            {/* Header with Start Next Round and Delete Round buttons */}
            <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <ExperimentOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
                        <Title level={2} style={{ margin: 0 }}>
                            Round {currentRound.round_number}
                            {currentRound.slate_seq_ids && (
                                <Text type="secondary" style={{ fontSize: '16px', fontWeight: 'normal', marginLeft: '12px' }}>
                                    - {currentRound.slate_seq_ids.split(',').length} mutants in slate
                                </Text>
                            )}
                        </Title>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        {isLastRound && currentRound.round_number > 1 && (
                            <Button
                                danger
                                size="large"
                                icon={<DeleteOutlined />}
                                onClick={handleDeleteRound}
                                title="Delete this round"
                            >
                                Delete Round
                            </Button>
                        )}

                        {!nextRoundExists && (
                            <Button
                                type="primary"
                                size="large"
                                disabled={!hasActivityData}
                                onClick={onStartNewRound}
                                title={!hasActivityData ? "Upload activity results to start next round" : "Start next round"}
                            >
                                Start Next Round
                            </Button>
                        )}
                    </div>
                </div>
                <Text type="secondary">
                    Started on {new Date(currentRound.date_started).toLocaleString()}
                </Text>
            </div>

            {renderWorkflowContent()}

            <SlateBuilder
                open={showSlateBuilder}
                onClose={() => setShowSlateBuilder(false)}
                onConfirm={handleSlateConfirm}
                seqIds={slateBuilderSeqIds}
                title="Build Slate for Campaign"
            />
        </div>
    );
};

export default CampaignRoundComponent;
