import fileDownload from "js-file-download";
import React, { useMemo, useState } from "react";
import {
    FaChevronLeft,
    FaChevronRight,
    FaClock,
    FaDownload,
    FaEye,
    FaFrownOpen,
    FaRedo,
    FaTrash,
} from "react-icons/fa";
import { getDockSdf, postDock } from "../../api/dockApi";
import { NewDockPrompt } from "../../util/newDockPrompt";
import { Dock, Invokation, DockInput } from "../../types/types";
import { notify } from "../../services/NotificationService";
import { TabContainer, DescriptionSection, TableSection, CollapsibleSection, ResponsiveTable } from "../../util/tabComponents";

interface DockTabProps {
    foldId: number;
    foldName: string | null;
    foldSequence: string | undefined;
    docks: Dock[] | null;
    jobs: Invokation[] | null;

    // UI Commands managed by the FoldView.
    displayedLigandNames: string[];
    ranks: { [ligandname: string]: number };
    displayLigandPose: (ligandName: string) => void;
    shiftFrame: (ligandName: string, shift: number) => void;
    deleteLigandPose: (ligandId: number, ligandName: string) => void;
}

type SortConfig = {
    key: keyof Dock | "fit" | null;
    direction: "ascending" | "descending";
};

const DockTab = React.memo((props: DockTabProps) => {
    const [sortConfig, setSortConfig] = useState<SortConfig>({
        key: "ligand_name",
        direction: "ascending",
    });
    const [showDockForm, setShowDockForm] = useState(false);

    const getDockState = (dock: Dock, jobs: Invokation[] | null) => {
        if (!jobs) return "queued";
        const job = jobs.find((invokation) => invokation.id === dock.invokation_id);
        return job?.state || "failed";
    };

    const downloadLigandPose = (ligandName: string) => {
        notify.info(`Downloading SDF file for ${ligandName}`);
        getDockSdf(props.foldId, ligandName).then(
            (sdf: Blob) => {
                if (!props.foldName) return;
                fileDownload(sdf, `${props.foldName}_${ligandName}.sdf`);
            },
            (error) => {
                notify.error(error.toString());
            }
        );
    };

    const rerunDock = (dock: Dock) => {
        const dockCopy: DockInput = { ...dock, fold_id: props.foldId };
        postDock(dockCopy).then(
            () => notify.success(`Successfully restarted docking for ${dock.ligand_name}`),
            (error) => notify.error(`Docking ${dock.ligand_name} failed: ${error}`)
        );
    };

    const getFit = (dock: Dock) => {
        if (dock.tool === "diffdock") {
            const confidenceStr =
                dock.pose_confidences?.split(",")[
                (props.ranks[dock.ligand_name] || 1) - 1
                ];
            return confidenceStr ? parseFloat(confidenceStr) : null;
        }
        return (props.ranks[dock.ligand_name] || 1) === 1 ? dock.pose_energy : null;
    };

    const compareValues = (
        key: keyof Dock | "fit",
        direction: "ascending" | "descending"
    ) => {
        return (a: Dock, b: Dock) => {
            let aValue, bValue;

            if (key === "fit") {
                if (a.tool !== b.tool) {
                    aValue = a.tool;
                    bValue = b.tool;
                } else {
                    aValue = Number(getFit(a));
                    bValue = Number(getFit(b));
                }
            } else {
                aValue = a[key];
                bValue = b[key];
            }

            if (aValue === bValue) return 0;
            if (aValue === null) return direction === "ascending" ? -1 : 1;
            if (bValue === null) return direction === "ascending" ? 1 : -1;
            return aValue < bValue
                ? direction === "ascending"
                    ? -1
                    : 1
                : direction === "ascending"
                    ? 1
                    : -1;
        };
    };

    const sortedDocks = useMemo(() => {
        if (!props.docks) return null;
        return [...props.docks].sort(
            compareValues(sortConfig.key || "ligand_name", sortConfig.direction)
        );
    }, [props.docks, sortConfig]);

    const requestSort = (key: keyof Dock | "fit") => {
        const direction =
            sortConfig.key === key && sortConfig.direction === "ascending"
                ? "descending"
                : "ascending";
        setSortConfig({ key, direction });
    };

    const getSortSymbol = (key: keyof Dock | "fit") => {
        return sortConfig.key === key
            ? sortConfig.direction === "ascending"
                ? " ↑"
                : " ↓"
            : "";
    };

    return (
        <TabContainer>
            {/* Description Section */}
            <DescriptionSection title="Small Molecule Docking">
                <p>
                    Use <a href="https://onlinelibrary.wiley.com/doi/pdf/10.1002/jcc.21334">Autodock Vina</a> or DiffDock to predict ligand poses. Sort and manage docking results or dock new ligands below.
                </p>
            </DescriptionSection>

            {/* Docking Results Table */}
            <TableSection title="Docking Results">
                <ResponsiveTable>
                    <thead>
                        <tr>
                            <th onClick={() => requestSort("ligand_name")}>
                                Name{getSortSymbol("ligand_name")}
                            </th>
                            <th onClick={() => requestSort("fit")}>
                                Fit{getSortSymbol("fit")}
                            </th>
                            <th>Rank</th>
                            <th onClick={() => requestSort("tool")}>
                                Tool{getSortSymbol("tool")}
                            </th>
                            <th>Bounding Box</th>
                            <th onClick={() => requestSort("ligand_smiles")}>
                                SMILES{getSortSymbol("ligand_smiles")}
                            </th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedDocks?.map((dock) => (
                            <tr key={dock.id}>
                                <td>{dock.ligand_name}</td>
                                <td>{getFit(dock)}</td>
                                <td>{props.ranks[dock.ligand_name]}</td>
                                <td>{dock.tool}</td>
                                <td>
                                    {dock.bounding_box_residue && dock.bounding_box_radius_angstrom
                                        ? `${dock.bounding_box_residue} (${dock.bounding_box_radius_angstrom} Å)`
                                        : "N/A"}
                                </td>
                                <td>
                                    <span
                                        style={{
                                            whiteSpace: "nowrap",
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                            display: "block",
                                            maxWidth: "200px",
                                        }}
                                        title={dock.ligand_smiles} // Tooltip with full SMILES
                                    >
                                        {dock.ligand_smiles}
                                    </span>
                                </td>
                                <td>
                                    {getDockState(dock, props.jobs) === "queued" ||
                                        getDockState(dock, props.jobs) === "running" ? (
                                        <FaClock
                                            uk-tooltip={`Docking is currently ${getDockState(dock, props.jobs)}`}
                                        />
                                    ) : getDockState(dock, props.jobs) === "failed" ? (
                                        <FaFrownOpen
                                            uk-tooltip="Docking failed. Consider rerunning this docking job."
                                        />
                                    ) : (
                                        <FaEye
                                            uk-tooltip="View this ligand's pose in the visualization pane."
                                            onClick={() => props.displayLigandPose(dock.ligand_name)}
                                        />
                                    )}
                                    {props.displayedLigandNames.includes(dock.ligand_name) && (
                                        <span>
                                            <FaChevronLeft
                                                uk-tooltip="View the previous pose prediction for this ligand."
                                                onClick={() => props.shiftFrame(dock.ligand_name, -1)}
                                            />
                                            <FaChevronRight
                                                uk-tooltip="View the next pose prediction for this ligand."
                                                onClick={() => props.shiftFrame(dock.ligand_name, 1)}
                                            />
                                        </span>
                                    )}
                                    <FaTrash
                                        uk-tooltip="Delete this docking result."
                                        onClick={() => props.deleteLigandPose(dock.id, dock.ligand_name)}
                                    />
                                    <FaRedo
                                        uk-tooltip="Rerun this docking job."
                                        onClick={() => rerunDock(dock)}
                                    />
                                    <FaDownload
                                        uk-tooltip="Download the SDF file for this ligand pose."
                                        onClick={() => downloadLigandPose(dock.ligand_name)}
                                    />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </ResponsiveTable>
            </TableSection>

            {/* Collapsible Dock New Ligands Section */}
            <CollapsibleSection
                title="Dock New Ligands"
                isOpen={showDockForm}
                onToggle={() => setShowDockForm(!showDockForm)}
            >
                <NewDockPrompt
                    foldIds={[props.foldId]}
                    existingLigands={{
                        [props.foldId]: (props.docks || []).map((dock) => dock.ligand_name),
                    }}
                />
            </CollapsibleSection>
        </TabContainer>
    );
});

export default DockTab;
