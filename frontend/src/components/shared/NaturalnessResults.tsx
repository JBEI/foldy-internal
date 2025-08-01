import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';
import Papa from 'papaparse';
import { CheckboxControl, NumberInputControl } from '../../util/controlComponents';
import ProposedSlateTable, { ProposedSlateTableColumn } from './ProposedSlateTable';
import { Selection } from '../FoldView/StructurePane';
import { notify } from '../../services/NotificationService';

// Define standard amino acid residues
const RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];

const NATURALNESS_COLUMN = 'probability';
const WT_MARGINAL_COLUMN = 'wt_marginal';

interface NaturalnessResultsProps {
    naturalnessCsvData: string;
    yamlConfig: string | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    runName?: string;
    onClose?: () => void;
    onBuildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
}

type RowData = {
    seqId: string;
    score: number;
    model: number | null;
}

const parseSeqId = (seqId: string): { wtResidue: string, locus: number, mutantResidue: string } => {
    if (seqId.includes('_')) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
    if (!match) {
        throw new Error(`Invalid seqId: "${seqId}"`);
    }
    return { wtResidue: match[1], locus: parseInt(match[2]), mutantResidue: match[3] };
}

const parseCsvDataIntoRowData = (naturalnessCsvDataString: string, useWtMarginalAsScore: boolean, zeroWildType: boolean, maxMutationsPerLocus: number | undefined, topPerformersToDisplay: number | undefined): RowData[] | null => {
    const { data, errors } = Papa.parse<Record<string, string>>(naturalnessCsvDataString, {
        header: true,
        delimiter: ',',
        skipEmptyLines: true,
        dynamicTyping: true
    });

    if (errors.length > 0) {
        notify.error(`Error parsing naturalness CSV: ${errors.map(error => error.message).join(', ')}`);
        return null;
    }

    const interiorTableRows = data.filter((row) => {
        const endsInSpecialCharacter = row['seq_id'].match(/.*<.*>/);
        const endsInDot = row['seq_id'].match(/.*\..*/);
        const endsInHyphen = row['seq_id'].match(/.*-.*/);
        const endsInBar = row['seq_id'].match(/.*\|.*/);
        if (endsInSpecialCharacter || endsInDot || endsInHyphen || endsInBar) {
            return false;
        }
        return true;
    }).map((row) => {
        var score;
        if (useWtMarginalAsScore) {
            score = parseFloat(row[WT_MARGINAL_COLUMN]);
            score = score ? Math.log(score || 1e-7) : null;
        } else {
            score = parseFloat(row[NATURALNESS_COLUMN]) || 0;
        }
        score = score || 0;

        const { wtResidue, locus, mutantResidue } = parseSeqId(row['seq_id']);

        if (zeroWildType && wtResidue == mutantResidue) {
            score = 0;
        }
        var model = null;
        if (row['model'] != null) {
            model = parseInt(row['model']);
        }

        return {
            seqId: row['seq_id'],
            score: score,
            model: model
        };
    });

    if (maxMutationsPerLocus !== undefined || topPerformersToDisplay !== undefined) {
        const filteredRows = new Map<number, RowData[]>();

        interiorTableRows.forEach(row => {
            const { locus } = parseSeqId(row.seqId);
            if (!filteredRows.has(locus)) {
                filteredRows.set(locus, []);
            }
            filteredRows.get(locus)!.push(row);
        });

        const finalRows: RowData[] = [];
        filteredRows.forEach((locusRows, locus) => {
            locusRows.sort((a, b) => b.score - a.score);
            const limit = maxMutationsPerLocus || locusRows.length;
            finalRows.push(...locusRows.slice(0, limit));
        });

        finalRows.sort((a, b) => b.score - a.score);
        const displayLimit = topPerformersToDisplay || finalRows.length;
        return finalRows.slice(0, displayLimit);
    }

    return interiorTableRows;
};

const NaturalnessTable: React.FC<{
    naturalnessCsvData: string;
    useWtMarginalAsScore: boolean;
    zeroWildType: boolean;
    maxMutationsPerLocus: number;
    topPerformersToDisplay: number;
    yamlConfig: string | null;
    setSelectedSubsequence: (selection: Selection | null) => void;
    onBuildSlate?: (seqIds: string[]) => void;
    disableSlateBuilder?: boolean;
}> = ({
    naturalnessCsvData,
    useWtMarginalAsScore,
    zeroWildType,
    maxMutationsPerLocus,
    topPerformersToDisplay,
    yamlConfig,
    setSelectedSubsequence,
    onBuildSlate,
    disableSlateBuilder,
}) => {
        if (!naturalnessCsvData) return null;

        const tableData: RowData[] | null = useMemo(() => {
            return parseCsvDataIntoRowData(naturalnessCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay);
        }, [naturalnessCsvData, useWtMarginalAsScore, zeroWildType, maxMutationsPerLocus, topPerformersToDisplay]);

        if (!tableData) return null;

        const columns: ProposedSlateTableColumn[] = [
            {
                key: "seqId",
                name: "Sequence ID",
                sortable: true,
                resizable: true,
                sortDescendingFirst: true
            },
            {
                key: "score",
                name: useWtMarginalAsScore ? "Log(WT Marginal Likelihood)" : "Probability",
                sortable: true,
                resizable: true,
                formatter: ({ row }: { row: any }) => row.score.toFixed(4),
                sortDescendingFirst: true
            }
        ];

        return (
            <ProposedSlateTable
                description="These mutants have the highest naturalness scores. Click on a sequence ID to highlight the residues on the structure."
                data={tableData}
                columns={columns}
                yamlConfig={yamlConfig}
                setSelectedSubsequence={setSelectedSubsequence}
                rowSelection={true}
                enableRowClick={true}
                showCopyButton={true}
                showHighlightButton={true}
                showHighlightOnModelButton={false}
                showSlateBuilderButton={!!onBuildSlate}
                disableSlateBuilderButton={disableSlateBuilder}
                onBuildSlate={onBuildSlate}
            />
        );
    };

const NaturalnessResults: React.FC<NaturalnessResultsProps> = ({
    naturalnessCsvData,
    yamlConfig,
    setSelectedSubsequence,
    runName = "Naturalness Results",
    onClose,
    onBuildSlate,
    disableSlateBuilder
}) => {
    const [maskWildType, setMaskWildType] = useState<boolean>(false);
    const [zeroWildType, setZeroWildType] = useState<boolean>(false);
    const [showWTMarginalLikelihood, setShowWTMarginalLikelihood] = useState<boolean>(true);
    const [maxMutationsPerLocus, setMaxMutationsPerLocus] = useState<number>(3);
    const [topPerformersToDisplay, setTopPerformersToDisplay] = useState<number>(24);

    const naturalnessPlot = useMemo(() => {
        if (!naturalnessCsvData) return null;

        const rowData = parseCsvDataIntoRowData(naturalnessCsvData, showWTMarginalLikelihood, zeroWildType, undefined, undefined);
        if (!rowData) return null;

        // Process data for heatmap
        const locusSet = new Set<number>();
        const scoreHeatmapData: { [key: string]: number } = {};
        const wtResidues: { [key: number]: string } = {};

        var ensembleMembers = 1.0;

        rowData.forEach(row => {
            if (row.model != null) {
                ensembleMembers = Math.max(ensembleMembers, row.model);
            }
            const seqId = row.seqId;

            const match = seqId.match(/([A-Z])(\d+)([A-Z])/);
            if (match) {
                const wtResidue = match[1];
                const locus = parseInt(match[2]);
                const mutantResidue = match[3];
                locusSet.add(locus);
                wtResidues[locus] = wtResidue;
                const key = `${locus}-${mutantResidue}`;
                if (key in scoreHeatmapData) {
                    scoreHeatmapData[key] += row.score;
                } else {
                    scoreHeatmapData[key] = row.score;
                }
            }
        });
        Object.keys(scoreHeatmapData).forEach(key => {
            scoreHeatmapData[key] /= ensembleMembers;
        });

        const loci = Array.from(locusSet).sort((a, b) => a - b);
        const zValues = RESIDUES.map(res =>
            loci.map(locus => {
                if (res === wtResidues[locus]) {
                    if (maskWildType) {
                        return null;
                    }
                }
                const key = `${locus}-${res}`;
                return key in scoreHeatmapData ? scoreHeatmapData[key] : null;
            })
        );
        const zmin = showWTMarginalLikelihood ? 0 : 0;
        const zmax = showWTMarginalLikelihood ? Math.max(...zValues.flat(2).filter(val => val !== null) as number[]) : 1;

        // Create customdata to match the z-values structure (RESIDUES x loci)
        const customData = RESIDUES.map(residue =>
            loci.map(locus => wtResidues[locus])
        );

        const hoverTemplate = showWTMarginalLikelihood ? '%{customdata}%{x}%{y}<br>Score: 10^%{z}<extra></extra>' : '%{customdata}%{x}%{y}<br>Probability: %{z}<extra></extra>';

        const plotlyData: Array<Partial<Data>> = [{
            type: 'heatmap',
            z: zValues,
            x: loci,
            y: RESIDUES,
            colorscale: 'RdYlBu_r',
            zmin: zmin,
            zmax: zmax,
            hovertemplate: hoverTemplate,
            customdata: customData,
        }];

        return (
            <div style={{ marginBottom: '20px', width: '100%', maxWidth: '100%', height: '400px', overflow: 'hidden' }}>
                <Plot
                    data={plotlyData}
                    layout={{
                        title: {
                            text: showWTMarginalLikelihood ? 'Log(WT Marginal Likelihood) Heatmap' : 'Naturalness Probability Heatmap',
                            font: { size: 14, color: '#262626' }
                        },
                        xaxis: {
                            title: { text: 'Locus', font: { size: 12, color: '#595959' } },
                            tickfont: { size: 10, color: '#8c8c8c' }
                        },
                        yaxis: {
                            title: { text: 'Amino Acid', font: { size: 12, color: '#595959' } },
                            tickfont: { size: 10, color: '#8c8c8c' }
                        },
                        autosize: true,
                        margin: { l: 50, r: 50, t: 50, b: 50 },
                        plot_bgcolor: 'white',
                        paper_bgcolor: 'white',
                        font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }
                    }}
                    useResizeHandler={true}
                    style={{ width: '100%', height: '100%' }}
                    config={{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }}
                />
            </div>
        );
    }, [naturalnessCsvData, showWTMarginalLikelihood, zeroWildType, maskWildType]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {onClose && (
                <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "10px"
                }}>
                    <h2 style={{ margin: 0, overflowWrap: 'anywhere' }}>
                        {runName}
                    </h2>
                    <button
                        onClick={onClose}
                        style={{
                            background: "none",
                            border: "none",
                            cursor: "pointer",
                            fontSize: "20px",
                            padding: "5px",
                            color: "#666"
                        }}
                        aria-label="Close"
                    >
                        ✕
                    </button>
                </div>
            )}
            <h3>Single Mutant Naturalness</h3>

            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '20px',
                marginBottom: '20px'
            }}>
                <div>
                    <NumberInputControl
                        label="Max number of mutants per locus"
                        value={maxMutationsPerLocus}
                        onChange={setMaxMutationsPerLocus}
                        min={1}
                    />
                    <NumberInputControl
                        label="Top mutants to display"
                        value={topPerformersToDisplay}
                        onChange={setTopPerformersToDisplay}
                        min={1}
                    />
                </div>
                <div>
                    <CheckboxControl
                        label="Mask wild-type amino acids"
                        checked={maskWildType}
                        onChange={setMaskWildType}
                    />
                    <CheckboxControl
                        label="Zero out wild-type amino acids"
                        checked={zeroWildType}
                        onChange={setZeroWildType}
                    />
                    <CheckboxControl
                        label="Display Log(WT Marginal Likelihood)"
                        checked={showWTMarginalLikelihood}
                        onChange={setShowWTMarginalLikelihood}
                    />
                </div>
            </div>

            <div>
                {naturalnessPlot}
            </div>

            <div>
                <h3>Proposed Slate</h3>
                <NaturalnessTable
                    naturalnessCsvData={naturalnessCsvData}
                    useWtMarginalAsScore={showWTMarginalLikelihood}
                    zeroWildType={zeroWildType}
                    maxMutationsPerLocus={maxMutationsPerLocus}
                    topPerformersToDisplay={topPerformersToDisplay}
                    yamlConfig={yamlConfig}
                    setSelectedSubsequence={setSelectedSubsequence}
                    onBuildSlate={onBuildSlate}
                    disableSlateBuilder={disableSlateBuilder}
                />
            </div>
        </div>
    );
};

export default NaturalnessResults;
