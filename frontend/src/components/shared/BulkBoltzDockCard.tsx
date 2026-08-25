import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
    Alert,
    Button,
    Card,
    Checkbox,
    Col,
    Form,
    Input,
    InputNumber,
    Modal,
    Progress,
    Row,
    Select,
    Space,
    Table,
    Tag,
    Tooltip,
    Typography,
} from 'antd';
import {
    ExperimentOutlined,
    PlusOutlined,
    ReloadOutlined,
    ThunderboltOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

import {
    BoltzDockBatch,
    BoltzDockBatchInput,
    BoltzDockEntry,
    createBoltzDockBatch,
    getBoltzDockBatch,
    getBoltzDockBatches,
    gradeBoltzDockBatch,
} from '../../api/boltzDockApi';
import type { Campaign, CampaignRound, Fold } from '../../types/types';
import { notify } from '../../services/NotificationService';

const { Text, Paragraph } = Typography;
const { TextArea } = Input;

interface BulkBoltzDockCardProps {
    campaign: Campaign;
    currentRound: CampaignRound;
    fold: Fold;
    activityData: Array<{ seq_id: string; activity: number }> | null;
}

interface BatchFormValues {
    name: string;
    variants: string;
    dockingMode: 'reaction' | 'ligands';
    ligands?: string;
    preComponents?: string;
    productName?: string;
    productSmiles?: string;
    diffusionSamples: number;
    msaMode: 'server' | 'reuse_source';
    p450Heme: boolean;
    proximalCysteine?: number;
}

const SUBSTRATE_A_SMILES = 'OC1=CC=C(CC)C=C1';
const SUBSTRATE_B_SMILES = 'NC1=CC=C(C(O)=O)C=C1';
const PRODUCT_PRESETS: Record<string, string> = {
    OrthoCC: 'OC1=CC=C(CC)C=C1C2=C(N)C=CC(C(O)=O)=C2',
    Ortho: 'OC1=CC=C(CC)C=C1NC2=CC=C(C(O)=O)C=C2',
    DC: 'OC1=CC=C(CC)C=C1C2=CC=C(N)C=C2',
};

const inferProductPreset = (campaignName: string): { name: string; smiles: string } | null => {
    const compactName = campaignName.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (compactName.includes('orthocc')) {
        return { name: 'OrthoCC', smiles: PRODUCT_PRESETS.OrthoCC };
    }
    if (compactName === 'dc' || compactName.includes('decarbox')) {
        return { name: 'DC', smiles: PRODUCT_PRESETS.DC };
    }
    if (compactName.includes('ortho')) {
        return { name: 'Ortho', smiles: PRODUCT_PRESETS.Ortho };
    }
    return null;
};

const formatMetric = (value: number | null | undefined, digits: number = 3) =>
    value === null || value === undefined ? '—' : value.toFixed(digits);

const statusColor = (state: string) => {
    if (state === 'finished') return 'success';
    if (state === 'failed') return 'error';
    if (state === 'started' || state === 'running') return 'processing';
    if (state === 'queued') return 'blue';
    return 'default';
};

const getProteinSequence = (fold: Fold): string | null => {
    const yamlSequence = fold.yaml_helper?.getProteinSequences()[0]?.[1];
    return yamlSequence || fold.sequence || null;
};

const detectP450Cysteine = (sequence: string | null): number | undefined => {
    if (!sequence) return undefined;
    const tailStart = Math.floor(sequence.length * 0.6);
    const tail = sequence.slice(tailStart);
    const motif = /F.{0,2}G.{1,7}C.G/g;
    let match: RegExpExecArray | null;
    let cysteine: number | undefined;
    while ((match = motif.exec(tail)) !== null) {
        cysteine = tailStart + match.index + match[0].lastIndexOf('C') + 1;
    }
    return cysteine;
};

const parseVariants = (value: string): string[] => {
    const variants = value
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
    return Array.from(new Set(variants));
};

const parseLigands = (value: string): Array<{ name: string; smiles: string }> => {
    return value
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line, index) => {
            const separator = line.indexOf(',');
            if (separator < 1 || separator === line.length - 1) {
                throw new Error(`Ligand line ${index + 1} must be name,SMILES.`);
            }
            return {
                name: line.slice(0, separator).trim(),
                smiles: line.slice(separator + 1).trim(),
            };
        });
};

const BulkBoltzDockCard: React.FC<BulkBoltzDockCardProps> = ({
    campaign,
    currentRound,
    fold,
    activityData,
}) => {
    const [form] = Form.useForm<BatchFormValues>();
    const [batches, setBatches] = useState<BoltzDockBatch[]>([]);
    const [selectedBatch, setSelectedBatch] = useState<BoltzDockBatch | null>(null);
    const [loading, setLoading] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [grading, setGrading] = useState(false);
    const [modalOpen, setModalOpen] = useState(false);
    const [p450Heme, setP450Heme] = useState(false);
    const [dockingMode, setDockingMode] = useState<'reaction' | 'ligands'>('reaction');

    const proteinSequence = useMemo(() => getProteinSequence(fold), [fold]);
    const detectedCysteine = useMemo(
        () => detectP450Cysteine(proteinSequence),
        [proteinSequence]
    );
    const defaultVariants = useMemo(() => {
        const seqIds = activityData?.map((row) => row.seq_id)
            || currentRound.slate_seq_ids?.split(',').filter(Boolean)
            || [];
        return ['WT', ...seqIds.filter((seqId) => seqId !== 'WT')].join('\n');
    }, [activityData, currentRound.slate_seq_ids]);

    const loadBatch = useCallback(async (batchId: number) => {
        setLoading(true);
        try {
            setSelectedBatch(await getBoltzDockBatch(batchId));
        } catch (error) {
            notify.error('Failed to load Boltz docking batch');
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, []);

    const loadBatches = useCallback(async () => {
        setLoading(true);
        try {
            const batchList = await getBoltzDockBatches({ campaignRoundId: currentRound.id });
            setBatches(batchList);
            const preferredId = batchList.some((batch) => batch.id === selectedBatch?.id)
                ? selectedBatch?.id
                : batchList[0]?.id;
            if (preferredId) {
                setSelectedBatch(await getBoltzDockBatch(preferredId));
            } else {
                setSelectedBatch(null);
            }
        } catch (error) {
            notify.error('Failed to load Boltz docking batches');
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, [currentRound.id, selectedBatch?.id]);

    useEffect(() => {
        loadBatches();
        // Reload when moving between campaign rounds; selection changes are handled explicitly.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentRound.id]);

    const openCreateModal = () => {
        const hasP450Motif = detectedCysteine !== undefined;
        const productPreset = inferProductPreset(campaign.name);
        setP450Heme(hasP450Motif);
        setDockingMode('reaction');
        form.setFieldsValue({
            name: `${campaign.name} round ${currentRound.round_number}`,
            variants: defaultVariants,
            dockingMode: 'reaction',
            ligands: '',
            preComponents: [
                `substrate_A,${SUBSTRATE_A_SMILES}`,
                `substrate_B,${SUBSTRATE_B_SMILES}`,
            ].join('\n'),
            productName: productPreset?.name,
            productSmiles: productPreset?.smiles,
            diffusionSamples: 3,
            msaMode: 'reuse_source',
            p450Heme: hasP450Motif,
            proximalCysteine: detectedCysteine,
        });
        setModalOpen(true);
    };

    const submitBatch = async (values: BatchFormValues) => {
        setSubmitting(true);
        try {
            const variants = parseVariants(values.variants);
            if (!variants.length) {
                throw new Error('Provide at least one variant.');
            }

            const request: BoltzDockBatchInput = {
                name: values.name,
                source_fold_id: campaign.fold_id,
                campaign_round_id: currentRound.id,
                variants,
                protein_chain_id: 'A',
                ligand_chain_id: 'C',
                diffusion_samples: values.diffusionSamples,
                msa_mode: values.msaMode,
                activities: activityData || undefined,
                tags: [`Campaign-${campaign.id}`, `Round-${currentRound.round_number}`],
                start_jobs: true,
            };
            if (values.dockingMode === 'reaction') {
                const components = parseLigands(values.preComponents || '');
                if (!components.length || !values.productName || !values.productSmiles) {
                    throw new Error('Provide pre-state components and one product.');
                }
                request.states = [
                    { name: 'pre_substrates', role: 'pre', components },
                    {
                        name: 'product',
                        role: 'post',
                        components: [{ name: values.productName, smiles: values.productSmiles }],
                    },
                ];
                request.comparisons = [{
                    name: 'pre_to_product',
                    pre_state: 'pre_substrates',
                    post_state: 'product',
                }];
            } else {
                const ligands = parseLigands(values.ligands || '');
                if (!ligands.length) {
                    throw new Error('Provide at least one ligand.');
                }
                request.ligands = ligands;
            }
            if (values.p450Heme || detectedCysteine !== undefined) {
                if (!values.proximalCysteine) {
                    throw new Error('A proximal cysteine residue is required for the P450 preset.');
                }
                request.cofactors = [{ chain_id: 'B', ccd: 'HEM' }];
                request.bonds = [{
                    atom1: ['A', values.proximalCysteine, 'SG'],
                    atom2: ['B', 1, 'FE'],
                }];
                request.pocket = {
                    contacts: [['B', 'FE']],
                    max_distance: 6,
                    force: true,
                };
            }

            const batch = await createBoltzDockBatch(request);
            setSelectedBatch(batch);
            setBatches((current) => [batch, ...current]);
            setModalOpen(false);
            notify.success(`Queued ${batch.entry_count} Boltz docking jobs`);
        } catch (error: any) {
            notify.error(
                error.response?.data?.message || error.message || 'Failed to create docking batch'
            );
        } finally {
            setSubmitting(false);
        }
    };

    const gradeBatch = async () => {
        if (!selectedBatch) return;
        setGrading(true);
        try {
            const updated = await gradeBoltzDockBatch(selectedBatch.id);
            setSelectedBatch(updated);
            notify.success(`Updated ${updated.graded_entries || 0} completed results`);
        } catch (error: any) {
            notify.error(error.response?.data?.message || 'Failed to grade docking batch');
        } finally {
            setGrading(false);
        }
    };

    const columns: ColumnsType<BoltzDockEntry> = [
        {
            title: 'Variant',
            dataIndex: 'seq_id',
            fixed: 'left',
            width: 190,
            render: (seqId: string, entry) => <Link to={`/fold/${entry.fold_id}`}>{seqId}</Link>,
        },
        {
            title: 'Docking state',
            dataIndex: 'ligand_name',
            width: 135,
            render: (name: string, entry) => (
                <Tooltip title={entry.state_data?.components.map((item) => item.name).join(' + ')}>
                    {name}
                </Tooltip>
            ),
        },
        {
            title: 'State',
            dataIndex: 'state',
            width: 105,
            render: (state: string, entry) => (
                <Tooltip title={entry.setup_error || entry.score_data?.grading_error || undefined}>
                    <Tag color={statusColor(state)}>{state}</Tag>
                </Tooltip>
            ),
        },
        {
            title: 'Activity',
            dataIndex: 'activity',
            width: 90,
            render: (value?: number | null) => formatMetric(value),
        },
        {
            title: (
                <Tooltip title="Rank within this ligand by Boltz ligand ipTM; this is pose confidence, not affinity.">
                    Pose rank
                </Tooltip>
            ),
            dataIndex: 'pose_quality_rank',
            width: 95,
            render: (value?: number | null) => value || '—',
        },
        {
            title: 'Ligand ipTM',
            width: 105,
            render: (_, entry) => formatMetric(entry.score_data?.ligand_iptm),
        },
        {
            title: 'Δ ipTM vs WT',
            width: 110,
            render: (_, entry) => formatMetric(entry.score_data?.delta_ligand_iptm_vs_wt),
        },
        {
            title: 'Ligand pLDDT',
            width: 115,
            render: (_, entry) => formatMetric(entry.score_data?.ligand_plddt),
        },
        {
            title: (
                <Tooltip title="For multi-component states, this is the farthest component's nearest-heavy-atom distance to the target; both substrates must be close.">
                    Target distance (Å)
                </Tooltip>
            ),
            width: 145,
            render: (_, entry) => formatMetric(entry.score_data?.target_distance, 2),
        },
        {
            title: 'Anchor distance (Å)',
            width: 145,
            render: (_, entry) => formatMetric(entry.score_data?.anchor_distance, 2),
        },
        {
            title: 'Pose RMSD (Å)',
            width: 125,
            render: (_, entry) => formatMetric(entry.score_data?.pose_rmsd, 2),
        },
    ];

    const terminalCount = selectedBatch
        ? (selectedBatch.state_counts.finished || 0) + (selectedBatch.state_counts.failed || 0)
        : 0;
    const progress = selectedBatch?.entry_count
        ? Math.round((terminalCount / selectedBatch.entry_count) * 100)
        : 0;
    const stateSummaryData = selectedBatch?.state_summaries
        ? Object.entries(selectedBatch.state_summaries).map(([stateName, summary]) => ({
            stateName,
            ...summary,
        }))
        : [];
    const comparisonData = selectedBatch?.comparison_data?.results || [];

    return (
        <Card
            title={(
                <Space>
                    <ThunderboltOutlined />
                    Boltz mutant docking
                </Space>
            )}
            style={{ marginTop: 24 }}
            extra={(
                <Space>
                    <Button icon={<ReloadOutlined />} onClick={loadBatches} loading={loading}>
                        Refresh
                    </Button>
                    <Button type="primary" icon={<PlusOutlined />} onClick={openCreateModal}>
                        New batch
                    </Button>
                </Space>
            )}
        >
            <Alert
                type="info"
                showIcon
                message="Structural grading, not binding free energy"
                description="The reaction mode docks all pre-state substrates simultaneously with the protein and cofactors, then docks the product separately. Conserved-atom RMSD is a pre/post pose proxy—not a transition-state energy, ΔG, or affinity estimate."
                style={{ marginBottom: 16 }}
            />

            {batches.length > 0 && (
                <Space wrap style={{ marginBottom: 16 }}>
                    <Text strong>Batch</Text>
                    <Select
                        style={{ minWidth: 260 }}
                        value={selectedBatch?.id}
                        options={batches.map((batch) => ({ value: batch.id, label: batch.name }))}
                        onChange={loadBatch}
                    />
                    <Button
                        icon={<ExperimentOutlined />}
                        onClick={gradeBatch}
                        loading={grading}
                        disabled={!selectedBatch}
                    >
                        Grade completed
                    </Button>
                </Space>
            )}

            {selectedBatch ? (
                <>
                    <Row gutter={[16, 8]} align="middle" style={{ marginBottom: 16 }}>
                        <Col xs={24} md={12}>
                            <Progress percent={progress} status={progress === 100 ? 'success' : 'active'} />
                        </Col>
                        <Col xs={24} md={12}>
                            <Text type="secondary">
                                {Object.entries(selectedBatch.state_counts)
                                    .map(([state, count]) => `${count} ${state}`)
                                    .join(' · ')}
                            </Text>
                        </Col>
                    </Row>
                    {stateSummaryData.length > 0 && (
                        <Table
                            rowKey="stateName"
                            size="small"
                            pagination={false}
                            style={{ marginBottom: 16 }}
                            dataSource={stateSummaryData}
                            columns={[
                                { title: 'Docking state', dataIndex: 'stateName' },
                                { title: 'Graded', dataIndex: 'graded_count' },
                                ...[
                                    ['ligand_iptm', 'ρ activity / ligand ipTM'],
                                    ['ligand_plddt', 'ρ activity / ligand pLDDT'],
                                    ['target_distance', 'ρ activity / target distance'],
                                    ['pose_rmsd', 'ρ activity / pose RMSD'],
                                ].map(([metric, title]) => ({
                                    title,
                                    render: (_: unknown, row: typeof stateSummaryData[number]) => {
                                        const result = row.metric_correlations[metric];
                                        return result
                                            ? `${formatMetric(result.spearman)} (n=${result.n})`
                                            : '—';
                                    },
                                })),
                            ]}
                            scroll={{ x: 950 }}
                        />
                    )}
                    {comparisonData.length > 0 && (
                        <Table
                            rowKey={(row) => `${row.comparison_name}-${row.seq_id}`}
                            size="small"
                            pagination={false}
                            style={{ marginBottom: 16 }}
                            dataSource={comparisonData}
                            columns={[
                                {
                                    title: 'Variant',
                                    dataIndex: 'seq_id',
                                    render: (seqId: string, row) => row.pre_fold_id
                                        ? <Link to={`/fold/${row.pre_fold_id}`}>{seqId}</Link>
                                        : seqId,
                                },
                                { title: 'Comparison', dataIndex: 'comparison_name' },
                                {
                                    title: 'Maintained-atom RMSD (Å)',
                                    render: (_, row) => formatMetric(row.maintained_atom_rmsd, 2),
                                },
                                {
                                    title: 'Mapped atoms',
                                    render: (_, row) => row.mapped_atom_count === undefined
                                        ? '—'
                                        : `${row.mapped_atom_count}/${row.pre_heavy_atom_count}`,
                                },
                                {
                                    title: 'Excluded atoms',
                                    render: (_, row) => {
                                        if (row.comparison_error) return row.comparison_error;
                                        if (!row.lost_atom_count) return 'None';
                                        return row.lost_atoms?.map((item) => (
                                            `${item.component}: ${item.atoms.map((atom) => atom.element).join(', ')}`
                                        )).join('; ');
                                    },
                                },
                                {
                                    title: 'Max displacement (Å)',
                                    render: (_, row) => formatMetric(
                                        row.max_maintained_atom_displacement,
                                        2
                                    ),
                                },
                            ]}
                            scroll={{ x: 900 }}
                        />
                    )}
                    <Table
                        rowKey="id"
                        columns={columns}
                        dataSource={selectedBatch.entries || []}
                        loading={loading}
                        size="small"
                        pagination={{ pageSize: 20, showSizeChanger: true }}
                        scroll={{ x: 1350 }}
                    />
                </>
            ) : (
                <Paragraph type="secondary">
                    Create a batch to expand variants × docking states into Boltz predictions.
                </Paragraph>
            )}

            <Modal
                title="New bulk Boltz docking batch"
                open={modalOpen}
                onCancel={() => setModalOpen(false)}
                onOk={() => form.submit()}
                okText="Create and queue"
                confirmLoading={submitting}
                width={760}
                destroyOnClose
            >
                <Form form={form} layout="vertical" onFinish={submitBatch}>
                    <Form.Item
                        name="name"
                        label="Batch name"
                        rules={[{ required: true, message: 'Enter a batch name' }]}
                    >
                        <Input />
                    </Form.Item>
                    <Form.Item name="dockingMode" label="Docking mode">
                        <Select
                            onChange={(value) => setDockingMode(value)}
                            options={[
                                {
                                    value: 'reaction',
                                    label: 'Pre/post reaction states (simultaneous substrates)',
                                },
                                { value: 'ligands', label: 'Independent single ligands' },
                            ]}
                        />
                    </Form.Item>
                    <Row gutter={16}>
                        <Col xs={24} md={12}>
                            <Form.Item
                                name="variants"
                                label="Variants"
                                tooltip="One Foldy seq_id per line. WT is included as a baseline."
                                rules={[{ required: true, message: 'Enter at least one variant' }]}
                            >
                                <TextArea rows={9} placeholder={'WT\nA42V\nA42V_L81F'} />
                            </Form.Item>
                        </Col>
                        <Col xs={24} md={12}>
                            {dockingMode === 'reaction' ? (
                                <Form.Item
                                    name="preComponents"
                                    label="Joint pre-state components"
                                    tooltip="Every name,SMILES line is placed in the same Boltz complex—not docked separately."
                                    rules={[{ required: true, message: 'Enter pre-state components' }]}
                                >
                                    <TextArea rows={9} />
                                </Form.Item>
                            ) : (
                                <Form.Item
                                    name="ligands"
                                    label="Independent ligands"
                                    tooltip="One name,SMILES pair per line; each becomes a separate job."
                                    rules={[{ required: true, message: 'Enter at least one ligand' }]}
                                >
                                    <TextArea
                                        rows={9}
                                        placeholder={'substrate_a,CC(=O)O\nsubstrate_b,c1ccccc1'}
                                    />
                                </Form.Item>
                            )}
                        </Col>
                    </Row>
                    {dockingMode === 'reaction' && (
                        <Row gutter={16}>
                            <Col xs={24} md={8}>
                                <Form.Item
                                    name="productName"
                                    label="Product name"
                                    rules={[{ required: true, message: 'Enter the product name' }]}
                                >
                                    <Input placeholder="OrthoCC" />
                                </Form.Item>
                            </Col>
                            <Col xs={24} md={16}>
                                <Form.Item
                                    name="productSmiles"
                                    label="Product SMILES"
                                    rules={[{ required: true, message: 'Enter the product SMILES' }]}
                                >
                                    <Input />
                                </Form.Item>
                            </Col>
                        </Row>
                    )}
                    <Row gutter={16}>
                        <Col xs={24} md={8}>
                            <Form.Item name="diffusionSamples" label="Diffusion samples">
                                <InputNumber min={1} max={10} style={{ width: '100%' }} />
                            </Form.Item>
                        </Col>
                        <Col xs={24} md={16}>
                            <Form.Item name="msaMode" label="MSA strategy">
                                <Select options={[
                                    {
                                        value: 'reuse_source',
                                        label: 'Reuse source MSA (mutate query row)',
                                    },
                                    { value: 'server', label: 'Build each MSA with server' },
                                ]} />
                            </Form.Item>
                        </Col>
                    </Row>
                    <Form.Item name="p450Heme" valuePropName="checked">
                        <Checkbox
                            disabled={detectedCysteine !== undefined}
                            onChange={(event) => setP450Heme(event.target.checked)}
                        >
                            P450 heme preset (include HEM, covalent Cys–Fe anchor, and Fe pocket target)
                        </Checkbox>
                    </Form.Item>
                    {p450Heme && (
                        <Form.Item
                            name="proximalCysteine"
                            label="Proximal cysteine residue"
                            rules={[{ required: true, message: 'Enter the heme-ligating cysteine' }]}
                            extra="Verify this 1-based residue index before queueing."
                        >
                            <InputNumber min={1} max={proteinSequence?.length} />
                        </Form.Item>
                    )}
                </Form>
            </Modal>
        </Card>
    );
};

export default BulkBoltzDockCard;
