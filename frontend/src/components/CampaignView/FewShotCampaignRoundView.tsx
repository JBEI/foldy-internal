import React, { useState, useEffect } from 'react';
import {
    Card,
    Typography,
    Button,
    Alert,
    Row,
    Col,
    Empty,
    Table,
    Upload,
    Form,
    Input,
    Select,
    Tooltip,
    Spin
} from 'antd';
import {
    UploadOutlined,
    PlusOutlined
} from '@ant-design/icons';
import type { Campaign, CampaignRound, Fold, FewShot } from '../../types/types';
import { updateCampaignRound, uploadCampaignRoundActivityFile, getCampaignRoundActivityData } from '../../api/campaignApi';
import { getFold } from '../../api/foldApi';
import { getFile } from '../../api/fileApi';
import { getFewShotDebugInfo } from '../../api/fewShotApi';
import { notify } from '../../services/NotificationService';
import { EmbeddingModal } from '../shared/EmbeddingModal';
import { FewShotModal } from '../shared/FewShotModal';
import FewShotMutantTable from '../shared/FewShotMutantTable';
import FewShotDebugPlots from '../shared/FewShotDebugPlots';
import MutantSlateCard from '../shared/MutantSlateCard';
import { getEmbeddingStatus, getNaturalnessStatus, getStatusDisplay, getFewShotStatus } from '../../util/statusHelpers';
import { Selection } from '../FoldView/StructurePane';

const { Title, Text, Paragraph } = Typography;

interface FewShotCampaignRoundViewProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    fold: Fold;
    activityData: Array<{ seq_id: string, activity: number }> | null;
    onRefresh: () => void;
    onRefreshRound?: () => void;
    buildSlate?: (seqIds: string[]) => void;
}

interface EmbeddingSelectionRow {
    key: string;
    requirement: string;
    availableEmbeddings: any[];
    selectedEmbeddingId?: number;
    newEmbeddingTemplate: any;
}

interface EmbeddingSelectionTableProps {
    title: string;
    description: string;
    rows: EmbeddingSelectionRow[];
    selectedEmbeddings: Record<string, number>;
    onEmbeddingChange: (rowKey: string, embeddingId: number | 'new', row?: EmbeddingSelectionRow) => void;
    maybeGetRequirementsErrorMessage?: () => string | null;
}

interface FewShotResultsContentProps {
    fewShotRun: FewShot;
    fold: Fold;
    campaign: Campaign;
    setSelectedSubsequence: (selection: Selection | null) => void;
    buildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
}

const FewShotResultsContent: React.FC<FewShotResultsContentProps> = ({
    fewShotRun,
    fold,
    campaign,
    setSelectedSubsequence,
    buildSlate,
    disableSlateBuilder
}) => {
    const [csvData, setCsvData] = useState<string | null>(null);
    const [debugData, setDebugData] = useState<any>(null);
    const [sortOptions, setSortOptions] = useState<{ [key: string]: string[] } | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadFewShotData = async () => {
            setLoading(true);
            try {
                // Load CSV data
                if (!fewShotRun.output_fpath) {
                    console.warn('No output path found for few-shot run:', fewShotRun);
                    return;
                }
                const csvBlob = await getFile(campaign.fold_id, fewShotRun.output_fpath);
                const csvText = await csvBlob.text();
                setCsvData(csvText);

                // Load debug data using helper function
                const { debugData, sortOptions } = await getFewShotDebugInfo(campaign.fold_id, fewShotRun);
                setDebugData(debugData);
                setSortOptions(sortOptions);
            } catch (error) {
                console.error('Error loading FewShot data:', error);
                notify.error(`Failed to load FewShot results: ${error}`);
            } finally {
                setLoading(false);
            }
        };

        loadFewShotData();
    }, [fewShotRun, campaign.fold_id]);

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                <Spin size="large" />
                <div style={{ marginTop: '16px' }}>
                    <Text type="secondary">Loading FewShot results...</Text>
                </div>
            </div>
        );
    }

    return (
        <div>
            {csvData && (
                <div style={{ marginBottom: '24px' }}>
                    <FewShotMutantTable
                        yamlConfig={fold?.yaml_config || null}
                        predictedMutantCsvData={csvData}
                        setSelectedSubsequence={setSelectedSubsequence}
                        sortOptions={sortOptions}
                        onBuildSlate={buildSlate}
                        disableSlateBuilder={disableSlateBuilder}
                    />
                </div>
            )}

            {debugData && (
                <div>
                    <h3>Training Metrics</h3>
                    <FewShotDebugPlots debugData={debugData} />
                </div>
            )}

            {!csvData && !debugData && (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Text type="secondary">No results data available</Text>
                </div>
            )}
        </div>
    );
};

const FewShotCampaignRoundView: React.FC<FewShotCampaignRoundViewProps> = ({
    campaign,
    currentRound,
    fold,
    activityData,
    onRefresh,
    onRefreshRound,
    buildSlate
}) => {
    // All hooks must be declared at the top level
    const [uploadingActivity, setUploadingActivity] = useState(false);
    const [selectedTemplateKeys, setSelectedTemplateKeys] = useState<React.Key[]>([]);
    const [customTemplates, setCustomTemplates] = useState<string>('');
    const [savingTemplates, setSavingTemplates] = useState(false);
    const [showEmbeddingModal, setShowEmbeddingModal] = useState(false);
    const [embeddingModalTemplate, setEmbeddingModalTemplate] = useState<any>(null);
    const [selectedEmbeddings, setSelectedEmbeddings] = useState<Record<string, number>>({});
    const [selectedNaturalnessRun, setSelectedNaturalnessRun] = useState<number | null>(null);
    const [showFewShotModal, setShowFewShotModal] = useState(false);
    const [fewShotTemplate, setFewShotTemplate] = useState<any>(null);
    const [priorRoundActivityData, setPriorRoundActivityData] = useState<Array<{ seq_id: string, activity: number }> | null>(null);

    // Rerun few-shot function
    const handleRerunFewShot = async () => {
        try {
            await updateCampaignRound(campaign.id!, currentRound.id, {
                few_shot_run_id: null
            });
            notify.success('Few-shot run cleared. The page will refresh.');
            onRefresh();
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to clear few-shot run');
        }
    };

    // All useEffect hooks must come before any conditional returns
    useEffect(() => {
        const fetchPriorRoundActivityData = async () => {
            if (currentRound.round_number <= 1) {
                setPriorRoundActivityData(null);
                return;
            }

            try {
                // Find the previous round
                const priorRound = campaign.rounds?.find(round =>
                    round.round_number === currentRound.round_number - 1
                );

                if (!priorRound || !priorRound.result_activity_fpath) {
                    console.warn('No prior round with activity data found');
                    setPriorRoundActivityData(null);
                    return;
                }

                const response = await getCampaignRoundActivityData(campaign.id!, priorRound.round_number);
                console.log('Prior round activity data:', response.data);
                setPriorRoundActivityData(response.data);
            } catch (error) {
                console.error('Failed to fetch prior round activity data:', error);
                setPriorRoundActivityData(null);
            }
        };

        fetchPriorRoundActivityData();
    }, [campaign.id, campaign.rounds, currentRound.round_number]);

    // Auto-select first available embedding for few-shot step 2
    useEffect(() => {
        if (!currentRound.input_templates || !fold?.embeddings) return;

        const allMatchingEmbeddings = fold.embeddings.filter(embedding =>
            embedding.embedding_model === campaign.embedding_model &&
            embedding.domain_boundaries === campaign.domain_boundaries
        );

        if (allMatchingEmbeddings.length === 0) return;

        const inputTemplates = currentRound.input_templates.split(',').map(t => t.trim());
        const selectedTemplates = ['WT', ...inputTemplates.filter(t => t !== 'WT')];

        const autoSelections: Record<string, number> = {};

        // Auto-select for naturalness warmstart
        const naturalnessWarmstartEmbeddings = allMatchingEmbeddings.filter(embedding =>
            embedding.dms_starting_seq_ids?.includes('WT')
        );
        if (naturalnessWarmstartEmbeddings.length > 0 && !selectedEmbeddings['naturalness-warmstart']) {
            autoSelections['naturalness-warmstart'] = naturalnessWarmstartEmbeddings[0].id;
        }

        // Auto-select for template embeddings
        selectedTemplates.forEach(template => {
            const key = `template-${template}`;
            const templateEmbeddings = allMatchingEmbeddings.filter(embedding =>
                embedding.dms_starting_seq_ids?.includes(template)
            );
            if (templateEmbeddings.length > 0 && !selectedEmbeddings[key]) {
                autoSelections[key] = templateEmbeddings[0].id;
            }
        });

        if (Object.keys(autoSelections).length > 0) {
            setSelectedEmbeddings(prev => ({
                ...prev,
                ...autoSelections
            }));
        }
    }, [fold?.embeddings, campaign.embedding_model, currentRound.input_templates, selectedEmbeddings]);

    // Step 2: Run Selection & Results - Auto-select naturalness run
    const matchingNaturalnessRuns = fold?.naturalness_runs?.filter(run =>
        run.logit_model === campaign.naturalness_model
    ) || [];

    useEffect(() => {
        if (matchingNaturalnessRuns.length > 0 && !selectedNaturalnessRun) {
            setSelectedNaturalnessRun(matchingNaturalnessRuns[0].id);
        }
    }, [matchingNaturalnessRuns, selectedNaturalnessRun]);

    // Variables and functions used throughout the component - defined before any conditional returns
    const allMatchingEmbeddings = fold?.embeddings?.filter(embedding =>
        embedding.embedding_model === campaign.embedding_model &&
        embedding.domain_boundaries === campaign.domain_boundaries
    ) || [];

    const inputTemplates = currentRound.input_templates?.split(',').map(t => t.trim()) || [];
    const selectedTemplates = ['WT', ...inputTemplates.filter(t => t !== 'WT')];

    // Helper function to get all templates from prior rounds
    const getAllAvailableTemplates = () => {
        const templates: Array<{ id: string, seq_id: string, round_number: number | null, source: string }> = [];

        // Always include WT at the top with round = None
        templates.push({
            id: 'wt-default',
            seq_id: 'WT',
            round_number: null,
            source: 'None'
        });

        if (!campaign.rounds) return templates;

        // Get unique templates from all prior rounds (not including current round)
        const seenTemplates = new Set(['WT']); // Already added WT

        campaign.rounds
            .filter(round => round.round_number < currentRound.round_number)
            .forEach(round => {
                if (round.promoted_templates) {
                    round.promoted_templates.forEach(templateId => {
                        // Skip WT since we already added it at the top
                        if (!seenTemplates.has(templateId)) {
                            templates.push({
                                id: `${round.round_number}-${templateId}`,
                                seq_id: templateId,
                                round_number: round.round_number,
                                source: `Round ${round.round_number}`
                            });
                            seenTemplates.add(templateId);
                        }
                    });
                }
            });

        return templates;
    };

    const handleActivityFileUpload = async (file: File) => {
        if (!campaign?.id) return false;

        setUploadingActivity(true);
        try {
            await uploadCampaignRoundActivityFile(campaign.id, currentRound.round_number, file);
            notify.success('Activity file uploaded successfully');
            onRefresh();
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to upload activity file');
        } finally {
            setUploadingActivity(false);
        }
        return false;
    };

    const handleTemplateToggle = async (seqId: string, isChecked: boolean) => {
        try {
            const currentTemplates = currentRound.promoted_templates || [];
            let updatedTemplates: string[];

            if (isChecked) {
                // Add to templates if not already there
                updatedTemplates = currentTemplates.includes(seqId)
                    ? currentTemplates
                    : [...currentTemplates, seqId];
            } else {
                // Remove from templates
                updatedTemplates = currentTemplates.filter(id => id !== seqId);
            }

            await updateCampaignRound(campaign.id!, currentRound.id, {
                promoted_templates: updatedTemplates
            });

            // Use targeted refresh instead of full page refresh
            if (onRefreshRound) {
                onRefreshRound();
            } else {
                onRefresh();
            }

            notify.success(isChecked ? 'Added to templates' : 'Removed from templates');
        } catch (error) {
            notify.error('Failed to update templates');
            console.error('Error updating templates:', error);
        }
    };


    const handleSaveTemplateSelection = async () => {
        setSavingTemplates(true);
        try {
            const availableTemplates = getAllAvailableTemplates();
            const selectedFromTable = selectedTemplateKeys.map(key => {
                const template = availableTemplates.find(t => t.id === key);
                return template?.seq_id;
            }).filter(Boolean);

            const customTemplateList = customTemplates
                .split(',')
                .map(t => t.trim())
                .filter(t => t.length > 0);

            const allSelectedTemplates = [...selectedFromTable, ...customTemplateList];
            const templatesCsv = allSelectedTemplates.join(',');

            await updateCampaignRound(campaign.id!, currentRound.id, {
                input_templates: templatesCsv
            });

            onRefresh();
            notify.success(`Selected ${allSelectedTemplates.length} templates for few-shot learning`);
        } catch (error) {
            notify.error('Failed to save template selection');
            console.error('Error saving templates:', error);
        } finally {
            setSavingTemplates(false);
        }
    };

    const handleEmbeddingModalClose = async () => {
        setShowEmbeddingModal(false);
        setEmbeddingModalTemplate(null);

        try {
            await getFold(campaign.fold_id);
            // Force refresh by calling onRefresh which should update fold data in parent
            onRefresh();
        } catch (error) {
            console.error('Error refreshing fold data:', error);
            notify.error('Failed to refresh fold data');
        }
    };

    // EmbeddingSelectionTable component - defined before conditional returns
    const EmbeddingSelectionTable: React.FC<EmbeddingSelectionTableProps> = ({
        title,
        description,
        rows,
        selectedEmbeddings,
        onEmbeddingChange,
        maybeGetRequirementsErrorMessage
    }) => {
        const getStatusForRow = (row: EmbeddingSelectionRow) => {
            const selectedEmbeddingId = selectedEmbeddings[row.key];

            if (!selectedEmbeddingId) {
                return { color: '#ff4d4f', text: 'Unsatisfied', icon: '✗' };
            }

            const selectedEmbedding = row.availableEmbeddings.find(e => e.id === selectedEmbeddingId);
            if (!selectedEmbedding) {
                return { color: '#ff4d4f', text: 'Unsatisfied', icon: '✗' };
            }

            const embeddingStatus = getEmbeddingStatus(selectedEmbedding, fold.jobs || null);
            const statusDisplay = getStatusDisplay(embeddingStatus);

            // If the embedding is complete, the requirement is satisfied
            if (statusDisplay.text === 'Complete') {
                return { color: statusDisplay.color, text: 'Satisfied', icon: statusDisplay.icon };
            }

            // Otherwise show the actual status (Running, Failed, etc.)
            return statusDisplay;
        };

        const columns = [
            {
                title: 'Requirement',
                dataIndex: 'requirement',
                key: 'requirement',
                width: '35%',
                ellipsis: true
            },
            {
                title: 'Embedding',
                key: 'embedding',
                width: '45%',
                render: (record: EmbeddingSelectionRow) => {
                    const truncateText = (text: string, maxLength: number = 25) => {
                        return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
                    };

                    const options = [
                        ...record.availableEmbeddings.map(embedding => {
                            const embeddingStatus = getEmbeddingStatus(embedding, fold.jobs || null);
                            const statusDisplay = getStatusDisplay(embeddingStatus);
                            const truncatedName = truncateText(embedding.name);
                            return {
                                value: embedding.id,
                                label: `${truncatedName} | ${statusDisplay.icon} ${statusDisplay.text}`,
                                title: `${embedding.name} | ${statusDisplay.icon} ${statusDisplay.text}` // Full name for hover tooltip
                            };
                        }),
                        {
                            value: 'new',
                            label: '+ Start New Embedding',
                            title: 'Start a new embedding run'
                        }
                    ];

                    const selectedOption = options.find(opt => opt.value === selectedEmbeddings[record.key]);
                    const hoverText = selectedOption ? selectedOption.title || selectedOption.label : '';

                    let startingValue: number | null = selectedEmbeddings[record.key];
                    // Check if starting value exists in options
                    if (startingValue && !options.find(opt => opt.value === startingValue)) {
                        // If not found, show error and return null
                        notify.error(`Embedding ${startingValue} is not a valid option. Setting to null.`);
                        startingValue = null;
                    }

                    return (
                        <div style={{ overflow: 'hidden' }}>
                            <Tooltip title={hoverText} placement="topLeft">
                                <Select
                                    style={{ width: '100%' }}
                                    placeholder="Select embedding..."
                                    value={startingValue}
                                    onChange={(value) => onEmbeddingChange(record.key, value, record)}
                                    options={options}
                                    showSearch
                                    optionFilterProp="label"
                                    dropdownStyle={{
                                        maxWidth: '600px'
                                    }}
                                />
                            </Tooltip>
                        </div>
                    );
                }
            },
            {
                title: 'Status',
                key: 'status',
                width: '20%',
                ellipsis: true,
                render: (record: EmbeddingSelectionRow) => {
                    const status = getStatusForRow(record);
                    return <span style={{ color: status.color }}>{status.icon} {status.text}</span>;
                }
            }
        ];

        return (
            <Card size="small" title={title}>
                {maybeGetRequirementsErrorMessage && maybeGetRequirementsErrorMessage() && (
                    <Alert
                        message="Requirements Error"
                        description={maybeGetRequirementsErrorMessage()}
                        type="error"
                        style={{ marginBottom: '16px' }}
                    />
                )}
                <Paragraph>{description}</Paragraph>
                <Table
                    dataSource={rows}
                    columns={columns}
                    rowKey="key"
                    pagination={false}
                    size="small"
                    tableLayout="fixed"
                />
            </Card>
        );
    };

    // Step 1: Template Selection - render conditionally instead of early return
    const shouldShowTemplateSelection = !currentRound.input_templates;

    if (shouldShowTemplateSelection) {
        const availableTemplates = getAllAvailableTemplates();

        return (
            <Card>
                <div style={{ marginBottom: '24px' }}>
                    <Title level={4}>Step 1: Select Protein Templates</Title>
                    <Paragraph>
                        Choose protein templates to use in the next round. The <strong>few-shot model</strong> will be used to <strong>evaluate all
                            possible single-mutants of the templates</strong> for testing in the next round.

                        If you want to keep your model only considering single mutants, you can select just the WT template.
                        If you want to stack only on the best performing mutant, you can just select the best performing mutant.

                        You can also add a sequence that wasn't tested in a prior round.
                    </Paragraph>
                </div>

                {availableTemplates.length > 0 ? (
                    <div style={{ marginBottom: '24px' }}>
                        <Text strong style={{ marginBottom: '16px', display: 'block' }}>
                            Available Templates from Prior Rounds:
                        </Text>
                        <Table
                            dataSource={availableTemplates}
                            columns={[
                                {
                                    title: 'Sequence ID',
                                    dataIndex: 'seq_id',
                                    key: 'seq_id',
                                    render: (seqId: string) => <Text code>{seqId}</Text>
                                },
                                {
                                    title: 'Source Round',
                                    dataIndex: 'source',
                                    key: 'source'
                                }
                            ]}
                            rowKey="id"
                            rowSelection={{
                                type: 'checkbox',
                                selectedRowKeys: selectedTemplateKeys,
                                onChange: setSelectedTemplateKeys,
                            }}
                            pagination={false}
                            size="small"
                        />
                    </div>
                ) : (
                    <Alert
                        message="No Templates Available"
                        description="No templates were found from previous rounds. You can still add custom templates below."
                        type="info"
                        style={{ marginBottom: '24px' }}
                    />
                )}

                <div style={{ marginBottom: '24px' }}>
                    <Text strong style={{ marginBottom: '8px', display: 'block' }}>
                        Custom Templates (comma-separated):
                    </Text>
                    <Input.TextArea
                        placeholder="Enter additional template sequences, separated by commas (e.g., WT, mutant1, mutant2)"
                        value={customTemplates}
                        onChange={(e) => setCustomTemplates(e.target.value)}
                        rows={3}
                        style={{ marginBottom: '16px' }}
                    />
                    <Text type="secondary">
                        These will be marked as "Custom" templates.
                    </Text>
                </div>

                <Button
                    type="primary"
                    size="large"
                    loading={savingTemplates}
                    onClick={handleSaveTemplateSelection}
                    disabled={selectedTemplateKeys.length === 0 && !customTemplates.trim()}
                >
                    Save Template Selection ({selectedTemplateKeys.length + (customTemplates.trim() ? customTemplates.split(',').filter(t => t.trim()).length : 0)} templates)
                </Button>
            </Card>
        );
    }

    const handleEmbeddingChange = (rowKey: string, embeddingId: number | 'new', row?: EmbeddingSelectionRow) => {
        if (embeddingId === 'new' && row) {
            setEmbeddingModalTemplate(row.newEmbeddingTemplate);
            setShowEmbeddingModal(true);
        } else if (typeof embeddingId === 'number') {
            setSelectedEmbeddings(prev => ({
                ...prev,
                [rowKey]: embeddingId
            }));
        }
    };

    // Create template FewShot object with pre-populated values
    const createTemplateFewShot = () => {
        // Get selected naturalness run
        const selectedNaturalnessRunData = matchingNaturalnessRuns.find(run =>
            run.id === selectedNaturalnessRun
        );

        // Get selected embedding paths (deduplicated)
        const embeddingPaths: string[] = [];
        const uniqueEmbeddingIds = Array.from(new Set(Object.values(selectedEmbeddings)));
        uniqueEmbeddingIds.forEach(embeddingId => {
            const embedding = allMatchingEmbeddings.find(e => e.id === embeddingId);
            if (embedding?.output_fpath) {
                embeddingPaths.push(embedding.output_fpath);
            }
        });

        console.log(`FewShot template has the following naturalness paths: ${selectedNaturalnessRunData?.output_fpath} for ${selectedNaturalnessRun}`);

        return {
            name: `${campaign.name}_R${currentRound.round_number}_fewshot`,
            mode: 'TorchMLPFewShotModel',
            num_mutants: 24,
            embedding_files: embeddingPaths.length > 0 ? embeddingPaths.join(',') : undefined,
            naturalness_files: selectedNaturalnessRunData?.output_fpath ? selectedNaturalnessRunData.output_fpath : undefined,
            finetuning_model_checkpoint: 'facebook/esm2_t6_8M_UR50D',
            params: `{
    "pretrain": true,
    "pretrain_epochs": 50,
    "ensemble_size": 5,
    "embedding_dim": 960,
    "hidden_dims": [100, 50],
    "dropout": 0.2,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "train_epochs": 200,
    "train_patience": 40,
    "val_frequency": 10,
    "do_validation_with_pair_fraction": 0.2,
    "decision_mode": "constantliar",
    "lie_noise_stddev_multiplier": 2.0
}`
        };
    };

    // Function to check if all requirements are satisfied
    const areAllRequirementsSatisfied = () => {
        // Check if naturalness run is selected and complete
        if (!selectedNaturalnessRun) return false;
        const selectedNaturalnessRunData = matchingNaturalnessRuns.find(run => run.id === selectedNaturalnessRun);
        if (!selectedNaturalnessRunData) return false;
        const naturalnessStatus = getNaturalnessStatus(selectedNaturalnessRunData, fold.jobs || null);
        const naturalnessStatusDisplay = getStatusDisplay(naturalnessStatus);
        if (naturalnessStatusDisplay.text !== 'Complete') return false;

        // Check if all embeddings are selected and complete
        const requiredEmbeddingKeys = [
            'naturalness-warmstart',
            ...selectedTemplates.map(template => `template-${template}`),
            'activity-measurements'
        ];

        for (const key of requiredEmbeddingKeys) {
            const selectedEmbeddingId = selectedEmbeddings[key];
            if (!selectedEmbeddingId) return false;

            const selectedEmbedding = allMatchingEmbeddings.find(e => e.id === selectedEmbeddingId);
            if (!selectedEmbedding) return false;

            const embeddingStatus = getEmbeddingStatus(selectedEmbedding, fold.jobs || null);
            const embeddingStatusDisplay = getStatusDisplay(embeddingStatus);
            if (embeddingStatusDisplay.text !== 'Complete') return false;
        }

        return true;
    };

    const handleLaunchFewShotModal = () => {
        const template = createTemplateFewShot();
        setFewShotTemplate(template);
        console.log('Launching few shot modal with template ', template);
        setShowFewShotModal(true);
    };

    const handleFewShotCreated = async (createdFewShot: any) => {
        try {
            // Update the campaign round with the created FewShot ID
            await updateCampaignRound(campaign.id!, currentRound.id, {
                few_shot_run_id: createdFewShot.id
            });

            // Refresh to get the updated campaign round
            onRefresh();
            notify.success(`FewShot run "${createdFewShot.name}" started successfully`);
        } catch (error) {
            notify.error('Failed to update campaign round with FewShot run');
            console.error('Error updating campaign round:', error);
            // Still refresh to see if it worked
            onRefresh();
        }
    };

    // If a FewShot run exists, show results instead of setup
    if (currentRound.few_shot_run) {
        const fewShotStatus = getFewShotStatus(currentRound.few_shot_run, fold.jobs || null);
        const statusDisplay = getStatusDisplay(fewShotStatus);
        const isComplete = statusDisplay.text === 'Complete';
        console.log('currentRound.few_shot_run', currentRound.few_shot_run);

        return (
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '20px',
                alignItems: 'start'
            }}>
                <Card
                    title={
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ color: statusDisplay.color }}>{statusDisplay.icon}</span>
                            <span>FewShot Results: {currentRound.few_shot_run.name}</span>
                        </div>
                    }
                >
                    {isComplete ? (
                        <FewShotResultsContent
                            fewShotRun={currentRound.few_shot_run}
                            fold={fold}
                            campaign={campaign}
                            setSelectedSubsequence={() => { }} // TODO: Implement if needed
                            buildSlate={buildSlate}
                            disableSlateBuilder={!!currentRound.slate_seq_ids}
                        />
                    ) : (
                        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                            <div style={{ fontSize: '48px', marginBottom: '16px' }}>
                                {statusDisplay.icon}
                            </div>
                            <Title level={4} style={{ color: statusDisplay.color }}>
                                {statusDisplay.text}
                            </Title>
                            <Text type="secondary">
                                FewShot run is {statusDisplay.text.toLowerCase()}. Please wait for it to complete.
                            </Text>
                            <div style={{ marginTop: '16px', display: 'flex', gap: '8px', justifyContent: 'center' }}>
                                <Button
                                    type="default"
                                    size="small"
                                    onClick={onRefresh}
                                >
                                    Refresh
                                </Button>
                                <Button
                                    type="default"
                                    size="small"
                                    onClick={handleRerunFewShot}
                                    disabled={!!currentRound.slate_seq_ids}
                                >
                                    Rerun few-shot algorithm
                                </Button>
                                <Button
                                    type="default"
                                    size="small"
                                    onClick={() => {
                                        // Get the invokation matching the few shot run
                                        const jobId = currentRound.few_shot_run?.invokation_id;
                                        if (jobId) {
                                            // Navigate to the logs tab with the specific job ID
                                            window.open(`/fold/${campaign.fold_id}/logs#logs_${jobId}`, '_blank');
                                        }
                                    }}
                                    disabled={!currentRound.few_shot_run?.id}
                                >
                                    Open Logs
                                </Button>
                            </div>
                        </div>
                    )}
                </Card>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    <MutantSlateCard
                        currentRound={currentRound}
                        activityData={activityData}
                        onActivityFileUpload={handleActivityFileUpload}
                        onTemplateToggle={handleTemplateToggle}
                        uploadingActivity={uploadingActivity}
                        updatingTemplates={false}
                        showActivityPlot={true}
                    />
                </div>
            </div>
        );
    }

    return (
        <div>
            <Card style={{ marginBottom: '20px' }}>
                <div style={{ marginBottom: '24px' }}>
                    <Title level={4}>Prepare Few-Shot Data</Title>
                    <Paragraph>
                        Configure the naturalness and embedding runs needed for few-shot predictions.
                    </Paragraph>
                </div>

                <Alert
                    message="Templates Selected"
                    description={`Using ${currentRound.input_templates?.split(',').length || 0} templates: ${currentRound.input_templates}`}
                    type="success"
                    style={{ marginBottom: '24px' }}
                    action={
                        <Button size="small" onClick={() => {
                            updateCampaignRound(campaign.id!, currentRound.id, { input_templates: '' })
                                .then(() => onRefresh())
                                .catch(() => notify.error('Failed to reset template selection'));
                        }}>
                            Change Templates
                        </Button>
                    }
                />
            </Card>

            <Row gutter={[20, 20]}>
                <Col sm={24} lg={12}>
                    <Card size="small" title="A) Naturalness Run">
                        <Paragraph>Select a naturalness run to use for few-shot predictions.</Paragraph>
                        {matchingNaturalnessRuns.length > 0 ? (
                            <Table
                                dataSource={[{
                                    key: 'naturalness-selection',
                                    requirement: 'Naturalness predictions',
                                    availableRuns: matchingNaturalnessRuns
                                }]}
                                columns={[
                                    {
                                        title: 'Requirement',
                                        dataIndex: 'requirement',
                                        key: 'requirement',
                                        width: '35%',
                                        ellipsis: true
                                    },
                                    {
                                        title: 'Naturalness Run',
                                        key: 'naturalness',
                                        width: '45%',
                                        render: () => {
                                            const truncateText = (text: string, maxLength: number = 25) => {
                                                return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
                                            };

                                            const options = matchingNaturalnessRuns.map(run => {
                                                const naturalnessStatus = getNaturalnessStatus(run, fold.jobs || null);
                                                const statusDisplay = getStatusDisplay(naturalnessStatus);
                                                const truncatedName = truncateText(run.name);
                                                return {
                                                    value: run.id,
                                                    label: `${truncatedName} | ${statusDisplay.icon} ${statusDisplay.text}`,
                                                    title: `${run.name} | ${statusDisplay.icon} ${statusDisplay.text}`
                                                };
                                            });

                                            const selectedOption = options.find(opt => opt.value === selectedNaturalnessRun);
                                            const hoverText = selectedOption ? selectedOption.title || selectedOption.label : '';

                                            return (
                                                <div style={{ overflow: 'hidden' }}>
                                                    <Tooltip title={hoverText} placement="topLeft">
                                                        <Select
                                                            style={{ width: '100%' }}
                                                            placeholder="Select naturalness run..."
                                                            value={selectedNaturalnessRun}
                                                            onChange={(value) => setSelectedNaturalnessRun(value)}
                                                            options={options}
                                                            showSearch
                                                            optionFilterProp="label"
                                                            dropdownStyle={{
                                                                maxWidth: '600px'
                                                            }}
                                                        />
                                                    </Tooltip>
                                                </div>
                                            );
                                        }
                                    },
                                    {
                                        title: 'Status',
                                        key: 'status',
                                        width: '20%',
                                        ellipsis: true,
                                        render: () => {
                                            if (!selectedNaturalnessRun) {
                                                return <span style={{ color: '#ff4d4f' }}>❌ Unsatisfied</span>;
                                            }

                                            const selectedRun = matchingNaturalnessRuns.find(run => run.id === selectedNaturalnessRun);
                                            if (!selectedRun) {
                                                return <span style={{ color: '#ff4d4f' }}>❌ Unsatisfied</span>;
                                            }

                                            const naturalnessStatus = getNaturalnessStatus(selectedRun, fold.jobs || null);
                                            const statusDisplay = getStatusDisplay(naturalnessStatus);

                                            // If complete, show satisfied
                                            if (statusDisplay.text === 'Complete') {
                                                return <span style={{ color: statusDisplay.color }}>{statusDisplay.icon} Satisfied</span>;
                                            }

                                            // Otherwise show actual status
                                            return <span style={{ color: statusDisplay.color }}>{statusDisplay.icon} {statusDisplay.text}</span>;
                                        }
                                    }
                                ]}
                                rowKey="key"
                                pagination={false}
                                size="small"
                                tableLayout="fixed"
                            />
                        ) : (
                            <Empty description="No matching naturalness runs" />
                        )}
                    </Card>
                </Col>
                <Col sm={24} lg={12}>
                    <EmbeddingSelectionTable
                        title="B) Embeddings for Naturalness Warm Start"
                        description="All single mutant variants of the WT sequence must be embedded for naturalness warm-start."
                        rows={[{
                            key: 'naturalness-warmstart',
                            requirement: 'Single mutant embeddings',
                            availableEmbeddings: allMatchingEmbeddings.filter(embedding =>
                                embedding.dms_starting_seq_ids?.includes('WT')
                            ),
                            newEmbeddingTemplate: {
                                name: `${campaign.name}_naturalness_warmstart`,
                                embedding_model: campaign.embedding_model,
                                dms_starting_seq_ids: 'WT',
                                extra_seq_ids: '',
                                domain_boundaries: campaign.domain_boundaries
                            }
                        }]}
                        selectedEmbeddings={selectedEmbeddings}
                        onEmbeddingChange={handleEmbeddingChange}
                    />
                </Col>

                <Col sm={24} lg={12}>
                    <EmbeddingSelectionTable
                        title="C) Embeddings for New Proteins"
                        description={`All candidate sequences, which are the single mutant variants of all ${selectedTemplates.length} templates, must be embedded in order to be evaluated.`}
                        rows={selectedTemplates.map(template => ({
                            key: `template-${template}`,
                            requirement: `Embeddings for ${template}`,
                            availableEmbeddings: allMatchingEmbeddings.filter(embedding =>
                                embedding.dms_starting_seq_ids?.includes(template)
                            ),
                            newEmbeddingTemplate: {
                                name: `${campaign.name}_template_${template}`,
                                embedding_model: campaign.embedding_model,
                                dms_starting_seq_ids: template,
                                extra_seq_ids: '',
                                domain_boundaries: campaign.domain_boundaries
                            }
                        }))}
                        selectedEmbeddings={selectedEmbeddings}
                        onEmbeddingChange={handleEmbeddingChange}
                    />
                </Col>

                <Col sm={24} lg={12}>
                    <EmbeddingSelectionTable
                        title="D) Embeddings for Activity Measurements"
                        description={`All measurements from prior round (Round ${currentRound.round_number - 1}) must be embedded in order to train the model.`}
                        rows={[{
                            key: 'activity-measurements',
                            requirement: 'Activity-based embeddings',
                            availableEmbeddings: (() => {
                                if (!priorRoundActivityData) return allMatchingEmbeddings;

                                // Get deduplicated sequence IDs from prior round activity data
                                const priorRoundSeqIds = Array.from(new Set(priorRoundActivityData.map(item => item.seq_id)));
                                const priorRoundSeqIdsSet = new Set(priorRoundSeqIds);

                                // Filter embeddings that have exactly these sequence IDs in extra_seq_ids
                                return allMatchingEmbeddings.filter(embedding => {
                                    if (!embedding.extra_seq_ids) return false;

                                    const embeddingSeqIds = new Set(embedding.extra_seq_ids.split(',').map(id => id.trim()).filter(id => id));

                                    // Check if the sets are exactly equal (same elements, same size)
                                    return embeddingSeqIds.size === priorRoundSeqIdsSet.size &&
                                        Array.from(priorRoundSeqIdsSet).every(id => embeddingSeqIds.has(id));
                                });
                            })(),
                            newEmbeddingTemplate: {
                                name: `${campaign.name}_R${currentRound.round_number - 1}_activity_measurements`,
                                embedding_model: campaign.embedding_model,
                                dms_starting_seq_ids: '',
                                extra_seq_ids: priorRoundActivityData ?
                                    Array.from(new Set(priorRoundActivityData.map(item => item.seq_id))).join(',') :
                                    '',
                                domain_boundaries: campaign.domain_boundaries
                            }
                        }]}
                        selectedEmbeddings={selectedEmbeddings}
                        onEmbeddingChange={handleEmbeddingChange}
                    />
                </Col>
            </Row>

            {/* FewShot Run Launcher Section */}
            <Card style={{ marginTop: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                    <span><strong>FewShot Activity Prediction</strong></span>
                </div>

                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                    <Title level={4}>
                        {areAllRequirementsSatisfied() ? 'Ready to Run FewShot Prediction' : 'Complete Requirements to Run FewShot Prediction'}
                    </Title>
                    <Text type="secondary" style={{ display: 'block', marginBottom: '32px' }}>
                        {areAllRequirementsSatisfied()
                            ? 'All requirements have been configured. Start the FewShot activity prediction to generate mutant recommendations.'
                            : 'Please complete all embedding selections and ensure the naturalness run is finished before starting FewShot prediction.'
                        }
                    </Text>

                    <Button
                        type="primary"
                        size="large"
                        icon={<PlusOutlined />}
                        onClick={handleLaunchFewShotModal}
                        disabled={!areAllRequirementsSatisfied()}
                        style={{
                            backgroundColor: '#1890ff',
                            fontSize: '16px',
                            height: '48px',
                            padding: '0 32px'
                        }}
                    >
                        Run FewShot Activity Prediction and Slate Builder
                    </Button>
                </div>
            </Card>


            <EmbeddingModal
                key={embeddingModalTemplate ? JSON.stringify(embeddingModalTemplate) : 'defaultEmbeddingModal'}
                open={showEmbeddingModal}
                onClose={handleEmbeddingModalClose}
                foldIds={[campaign.fold_id]}
                title="Start Embedding Run for Few-Shot"
                templateEmbedding={embeddingModalTemplate}
            />

            <FewShotModal
                key={fewShotTemplate ? JSON.stringify(fewShotTemplate) : 'defaultFewShotModal'}
                open={showFewShotModal}
                onClose={(createdFewShot?: any) => {
                    setShowFewShotModal(false);
                    setFewShotTemplate(null);

                    // If a FewShot was created, update the campaign round with its ID
                    console.log('Closing the few shot modal with createdFewShot ', createdFewShot);
                    if (createdFewShot) {
                        handleFewShotCreated(createdFewShot);
                    } else {
                        onRefresh(); // Fallback to full refresh
                    }
                }}
                foldId={campaign.fold_id}
                files={null} // Files not available in current Fold type
                evolutions={fold.few_shots || null}
                campaignRounds={campaign.rounds || null}
                title={`FewShot Run - ${campaign.name} Round ${currentRound.round_number}`}
                templateFewShot={fewShotTemplate}
                defaultActivityFileSource="campaign"
                fewShotCampaignRoundIdForActivityFile={(() => {
                    // Find the previous round with activity data
                    const priorRound = campaign.rounds?.find(round =>
                        round.round_number === currentRound.round_number - 1 && round.result_activity_fpath
                    );
                    return priorRound?.id;
                })()}
            />
        </div>
    );
};

export default FewShotCampaignRoundView;
