import React, { useEffect, useRef, useState } from 'react';
import { Modal, Alert, Button as AntButton, Typography, Upload } from 'antd';
import { QuestionCircleOutlined, UploadOutlined } from '@ant-design/icons';
import { startLogits } from '../../api/embedApi';
import { getFile, getFileList } from '../../api/fileApi';
import { notify } from '../../services/NotificationService';
import { ESMModelPicker } from '../FoldView/ESMModelPicker';
import { FormRow, FormField } from '../../util/tabComponents';
import { TextInputControl, CheckboxControl } from '../../util/controlComponents';
import { FileInfo, Logit } from '../../types/types';
import { findPreexistingMsaPath } from '../../util/msaContext';

const { Text, Paragraph, Title } = Typography;

interface NaturalnessModalProps {
    open: boolean;
    onClose: () => void;
    foldIds: number[];
    files?: FileInfo[];
    title?: string;
    templateNaturalnessRun?: Logit;
}

export const NaturalnessModal: React.FC<NaturalnessModalProps> = ({
    open,
    onClose,
    foldIds,
    files,
    title = "New Naturalness Run",
    templateNaturalnessRun
}) => {
    const [runName, setRunName] = useState<string>(templateNaturalnessRun?.name || '');
    const [logitModel, setLogitModel] = useState<string>(templateNaturalnessRun?.logit_model || 'esmc_600m');
    const [useStructure, setUseStructure] = useState<boolean>(templateNaturalnessRun?.use_structure || false);
    const [getDepthTwoLogits, setGetDepthTwoLogits] = useState<boolean>(templateNaturalnessRun?.get_depth_two_logits || false);
    const [useMsaContext, setUseMsaContext] = useState<boolean>(templateNaturalnessRun?.use_msa_context || false);
    const [msaA3m, setMsaA3m] = useState<string | null>(null);
    const [msaFile, setMsaFile] = useState<File | null>(null);
    const [autoMsaPath, setAutoMsaPath] = useState<string | null>(null);
    const [autoMsaLoading, setAutoMsaLoading] = useState<boolean>(false);
    const [showHelpModal, setShowHelpModal] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const autoMsaAttemptedRef = useRef<string | null>(null);
    const isE1Model = logitModel.startsWith('e1_');
    const isBulkRun = foldIds.length > 1;
    const preexistingMsaPath = templateNaturalnessRun?.msa_a3m_path || findPreexistingMsaPath(files);

    type BulkRunResult = {
        foldId: number;
        status: 'started' | 'skipped' | 'failed';
        error?: string;
    };

    useEffect(() => {
        if (!isE1Model && useMsaContext) {
            setUseMsaContext(false);
            setMsaA3m(null);
            setMsaFile(null);
            setAutoMsaPath(null);
            setAutoMsaLoading(false);
            autoMsaAttemptedRef.current = null;
        }
    }, [isE1Model, useMsaContext]);

    useEffect(() => {
        if (!useMsaContext) {
            setAutoMsaPath(null);
            setAutoMsaLoading(false);
            autoMsaAttemptedRef.current = null;
        }
    }, [useMsaContext]);

    useEffect(() => {
        if (!open || !useMsaContext || !isE1Model) {
            return;
        }
        if (!preexistingMsaPath || foldIds.length !== 1) {
            return;
        }
        if (msaA3m || msaFile) {
            return;
        }
        if (autoMsaAttemptedRef.current === preexistingMsaPath) {
            return;
        }

        autoMsaAttemptedRef.current = preexistingMsaPath;
        setAutoMsaPath(preexistingMsaPath);
        setAutoMsaLoading(true);
        let cancelled = false;

        const filePath = preexistingMsaPath.startsWith('/') ? preexistingMsaPath.slice(1) : preexistingMsaPath;
        getFile(foldIds[0], filePath)
            .then((fileBlob) => fileBlob.text())
            .then((content) => {
                if (cancelled) return;
                setMsaA3m(content);
                setAutoMsaLoading(false);
            })
            .catch((error) => {
                if (cancelled) return;
                setAutoMsaLoading(false);
                notify.error(`Failed to load existing MSA: ${error}`);
            });

        return () => {
            cancelled = true;
        };
    }, [open, useMsaContext, isE1Model, msaA3m, msaFile, preexistingMsaPath, foldIds]);

    const getErrorMessage = (error: unknown): string => {
        if (error instanceof Error) {
            return error.message;
        }
        return String(error);
    };

    const resolveMsaA3mForFold = async (foldId: number): Promise<string | null> => {
        const fileList = await getFileList(foldId);
        const preexistingPath = findPreexistingMsaPath(fileList);
        if (!preexistingPath) {
            return null;
        }
        const filePath = preexistingPath.startsWith('/') ? preexistingPath.slice(1) : preexistingPath;
        const fileBlob = await getFile(foldId, filePath);
        return fileBlob.text();
    };

    const handleStartLogit = async () => {
        if (!runName.trim()) {
            notify.error('Run name is required.');
            return;
        }
        if (useMsaContext && !isE1Model) {
            notify.error('MSA context is only supported for E1 models.');
            return;
        }
        if (useMsaContext && !msaA3m && !isBulkRun) {
            notify.error(autoMsaLoading ? 'Loading existing MSA, please wait.' : 'Please upload a .a3m MSA file.');
            return;
        }

        setIsLoading(true);

        try {
            const results: BulkRunResult[] = await Promise.all(
                foldIds.map(async (foldId) => {
                    const startRun = async (msaContents: string | null) => {
                        await startLogits(
                            foldId,
                            runName,
                            logitModel,
                            useStructure,
                            getDepthTwoLogits,
                            useMsaContext,
                            msaContents
                        );
                    };

                    if (!useMsaContext || !isBulkRun || msaA3m) {
                        try {
                            await startRun(msaA3m);
                            return { foldId, status: 'started' };
                        } catch (error) {
                            return { foldId, status: 'failed', error: getErrorMessage(error) };
                        }
                    }

                    try {
                        await startRun(null);
                        return { foldId, status: 'started' };
                    } catch (error) {
                        const message = getErrorMessage(error);
                        const missingMsa = message.toLowerCase().includes('msa_a3m is required');
                        if (!missingMsa) {
                            return { foldId, status: 'failed', error: message };
                        }

                        try {
                            const resolvedMsa = await resolveMsaA3mForFold(foldId);
                            if (!resolvedMsa) {
                                return { foldId, status: 'skipped' };
                            }
                            await startRun(resolvedMsa);
                            return { foldId, status: 'started' };
                        } catch (retryError) {
                            return { foldId, status: 'failed', error: getErrorMessage(retryError) };
                        }
                    }
                })
            );

            const started = results.filter(result => result.status === 'started');
            const skipped = results.filter(result => result.status === 'skipped');
            const failed = results.filter(result => result.status === 'failed');

            if (started.length > 0) {
                notify.success(`Started naturalness runs for ${started.length} fold(s)`);
            }
            if (skipped.length > 0) {
                notify.warning(`Skipped ${skipped.length} fold(s) without available MSA files.`);
            }
            if (failed.length > 0) {
                notify.error(`Failed to start ${failed.length} fold(s): ${failed[0].error ?? 'Unknown error'}`);
            }
            if (started.length === 0 && skipped.length > 0 && failed.length === 0) {
                notify.warning('No naturalness runs started. Provide MSA files or disable MSA context.');
            }

            if (started.length > 0) {
                // Reset form
                setRunName('');
                setLogitModel('esmc_600m');
                setUseStructure(false);
                setGetDepthTwoLogits(false);
                setUseMsaContext(false);
                setMsaA3m(null);
                setMsaFile(null);
                setAutoMsaPath(null);
                setAutoMsaLoading(false);
                autoMsaAttemptedRef.current = null;

                onClose();
            }
        } catch (error) {
            notify.error(`Failed to start naturalness runs: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleMsaFileChange = (file: File | null) => {
        setMsaFile(file);
        setAutoMsaLoading(false);
        if (file) {
            setAutoMsaPath(null);
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target?.result as string;
                setMsaA3m(content);
            };
            reader.readAsText(file);
        } else {
            setMsaA3m(null);
            autoMsaAttemptedRef.current = null;
        }
    };

    return (
        <>
            <Modal
                title={title}
                open={open}
                onCancel={onClose}
                footer={[
                    <AntButton key="cancel" onClick={onClose}>
                        Cancel
                    </AntButton>,
                    <AntButton
                        key="start"
                        type="primary"
                        onClick={handleStartLogit}
                        disabled={!runName.trim()}
                        loading={isLoading}
                    >
                        Start Naturalness Run{foldIds.length > 1 ? 's' : ''}
                    </AntButton>
                ]}
                width={600}
            >
                {/* Help Alert */}
                <Alert
                    message="What is Naturalness?"
                    description={
                        <div>
                            <Paragraph>
                                Naturalness uses protein language models to score how "natural" each possible amino acid mutation looks.
                                Higher scores indicate mutations that are more likely to maintain protein function.
                            </Paragraph>
                            <AntButton
                                type="link"
                                icon={<QuestionCircleOutlined />}
                                onClick={() => setShowHelpModal(true)}
                                style={{ padding: 0 }}
                            >
                                View detailed naturalness guide
                            </AntButton>
                        </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: '20px' }}
                />

                <div style={{ marginBottom: '16px' }}>
                    <Text strong>Target Folds:</Text> {foldIds.length} fold{foldIds.length > 1 ? 's' : ''}
                </div>

                <FormRow>
                    <FormField>
                        <TextInputControl
                            label="Name"
                            value={runName}
                            onChange={setRunName}
                            placeholder="Enter run name"
                        />
                    </FormField>

                    <FormField>
                        <ESMModelPicker
                            value={logitModel}
                            onChange={setLogitModel}
                        />
                    </FormField>
                </FormRow>

                <FormRow>
                    <FormField>
                        <CheckboxControl
                            label="Use Structure (experimental)"
                            checked={useStructure}
                            onChange={setUseStructure}
                        />
                    </FormField>

                    <FormField>
                        <CheckboxControl
                            label="Get Depth Two Logits (experimental)"
                            checked={getDepthTwoLogits}
                            onChange={setGetDepthTwoLogits}
                        />
                    </FormField>
                </FormRow>

                {isE1Model && (
                    <div style={{ marginTop: '8px' }}>
                        <CheckboxControl
                            label="Use E1 MSA context (.a3m)"
                            checked={useMsaContext}
                            onChange={setUseMsaContext}
                        />
                        {useMsaContext && (
                            <div style={{ marginTop: '8px' }}>
                                <Typography.Text strong style={{ marginBottom: '8px', display: 'block' }}>
                                    MSA (.a3m) File
                                </Typography.Text>
                                <Upload
                                    beforeUpload={(file) => {
                                        handleMsaFileChange(file);
                                        return false;
                                    }}
                                    accept=".a3m"
                                    maxCount={1}
                                    fileList={msaFile ? [{
                                        uid: '1',
                                        name: msaFile.name,
                                        status: 'done'
                                    }] : []}
                                    onRemove={() => handleMsaFileChange(null)}
                                >
                                    <AntButton icon={<UploadOutlined />}>
                                        Select .a3m File
                                    </AntButton>
                                </Upload>
                                {isBulkRun && (
                                    <Typography.Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
                                        Batch mode: leave blank to use each fold's stored or existing MSA file, or upload once to apply to all folds.
                                    </Typography.Text>
                                )}
                                {autoMsaPath && !msaFile && (
                                    <Typography.Text type="secondary" style={{ display: 'block', marginTop: '8px' }}>
                                        Using existing MSA from {autoMsaPath}{autoMsaLoading ? ' (loading...)' : ''}. Upload a file to override.
                                    </Typography.Text>
                                )}
                                {msaA3m && (
                                    <div style={{
                                        marginTop: '8px',
                                        padding: '8px',
                                        backgroundColor: '#f5f5f5',
                                        border: '1px solid #d9d9d9',
                                        borderRadius: '4px',
                                        fontSize: '11px',
                                        fontFamily: 'monospace',
                                        color: '#666',
                                        maxHeight: '120px',
                                        overflow: 'hidden'
                                    }}>
                                        <div style={{ marginBottom: '4px', fontSize: '10px', fontWeight: 500, textTransform: 'uppercase' }}>
                                            MSA Preview (first 5 lines)
                                        </div>
                                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                                            {msaA3m.split('\n').slice(0, 5).join('\n')}
                                            {msaA3m.split('\n').length > 5 && '\n...'}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </Modal>

            {/* Detailed Help Modal */}
            <Modal
                title="Naturalness Analysis Guide"
                open={showHelpModal}
                onCancel={() => setShowHelpModal(false)}
                footer={[
                    <AntButton key="close" onClick={() => setShowHelpModal(false)}>
                        Close
                    </AntButton>
                ]}
                width={700}
            >
                <div>
                    <Title level={4}>What is Naturalness?</Title>
                    <Paragraph>
                        Naturalness analysis uses protein language models (PLMs) to evaluate how "natural" or likely
                        each possible amino acid substitution appears based on evolutionary patterns learned from
                        millions of protein sequences.
                    </Paragraph>

                    <Title level={4}>How to Use</Title>
                    <ul>
                        <li><Text strong>Model Selection:</Text> Choose from different PLMs (ESM-C models recommended)</li>
                        <li><Text strong>Structure Integration:</Text> Optionally include 3D structure information</li>
                        <li><Text strong>Depth Two Logits:</Text> Advanced option for pair mutation analysis</li>
                    </ul>

                    <Title level={4}>Interpreting Results</Title>
                    <Paragraph>
                        The heatmap shows naturalness scores for each position-residue combination:
                    </Paragraph>
                    <ul>
                        <li><Text strong>Higher scores:</Text> More "natural" mutations, likely to preserve function</li>
                        <li><Text strong>Lower scores:</Text> Less natural mutations, may disrupt protein</li>
                        <li><Text strong>Wild-type masking:</Text> Option to hide original residues for clearer visualization</li>
                    </ul>

                    <Alert
                        message="Estimated Cost"
                        description="~$1 per naturalness run"
                        type="success"
                        showIcon
                        style={{ marginTop: '16px' }}
                    />

                    <Paragraph style={{ marginTop: '16px' }}>
                        Results can be downloaded as CSV files containing naturalness scores for all single mutations.
                    </Paragraph>
                </div>
            </Modal>
        </>
    );
};
