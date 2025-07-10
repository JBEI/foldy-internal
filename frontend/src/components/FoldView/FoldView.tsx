import jquery from "jquery";
import React, { Component } from "react";
import { useParams, useNavigate } from "react-router-dom";
import UIkit from "uikit";
import { Button, Card, Typography } from 'antd';

const { Title } = Typography;
import { notify } from "../../services/NotificationService";
import { queueJob } from "../../api/commonApi";
import { deleteDock } from "../../api/dockApi";
import { getFoldPfam, getInvokation } from "../../api/foldApi";
import DockTab from "./DockTab";
import "./FoldView.scss";
import PaeTab from "./PaeTab";
import JobsTab from "./JobsTab";
import SequenceTab from "./SequenceTab";
import fileDownload from "js-file-download";
import NaturalnessTab from "./NaturalnessTab";
import EmbedTab from "./EmbedTab";
import EvolveTab from "./EvolveTab";
import { Annotations, RenderableAnnotations, FileInfo, Fold, Invokation } from "../../types/types";
import { getFile, getFileList } from "../../api/fileApi";
import { getFold, updateFold } from "../../api/foldApi";
import StructurePane, { Selection } from "./StructurePane";
import FileTab from "./FileTab";

const REFRESH_STATE_PERIOD = 5000;
const REFRESH_STATE_MAX_ITERS = 200;


interface FoldProps {
    foldId: number;
    userType: string | null;
    navigate: ReturnType<typeof useNavigate>;
}


interface FoldState {
    foldData: Fold | null;

    // Note that a subset of job data is also in foldData.
    files: FileInfo[];
    jobs: Invokation[] | null;
    cifString: string | null;
    pdbString: string | null;

    renderablePfamAnnotations: RenderableAnnotations | null;

    structureFailedToLoad: boolean;
    paeIsOnScreen: boolean;
    contactIsOnScreen: boolean;
    showSplitScreen: boolean;
    numRefreshes: number;

    selectedSubsequence: Selection | null;
    currentFolderPath: string;
}

// From UIkit's definition of a "medium" window: https://getuikit.com/docs/visibility
const WINDOW_WIDTH_FOR_SPLIT_SCREEN = 960;
const MAX_JOBS_TO_REFRESH = 5;
class InternalFoldView extends Component<FoldProps, FoldState> {
    interval: NodeJS.Timeout | null = null;

    constructor(props: FoldProps) {
        super(props);

        this.state = {
            foldData: null,
            files: [],
            jobs: null,
            cifString: null,
            pdbString: null,

            renderablePfamAnnotations: null,

            structureFailedToLoad: false,
            paeIsOnScreen: false,
            contactIsOnScreen: false,
            showSplitScreen: window.innerWidth >= WINDOW_WIDTH_FOR_SPLIT_SCREEN,
            numRefreshes: 0,

            selectedSubsequence: null,
            currentFolderPath: '/',
        };
    }

    preventDefault = (e: any) => e.preventDefault();

    handleResize = () => {
        const newShowSplitScreen =
            window.innerWidth >= WINDOW_WIDTH_FOR_SPLIT_SCREEN;
        if (newShowSplitScreen !== this.state.showSplitScreen) {
            this.setState({ showSplitScreen: newShowSplitScreen });
        }
    };

    openUpLogsForJob = (jobId?: number) => {
        const tabElement = document.getElementById('tab');
        if (tabElement) {
            // 1 is the index of the Logs tab
            UIkit.tab(tabElement).show(1);

            // If a jobId is provided, we can add logic to scroll to that specific job
            if (jobId && this.state.jobs) {
                // Add a small delay to ensure the tab has switched
                setTimeout(() => {
                    const jobElement = document.getElementById(`logs_${jobId.toString()}`);
                    if (jobElement) {
                        jobElement.scrollIntoView({ behavior: 'smooth' });
                    }
                }, 100);
            }
        }
    }

    refreshFoldDataFromBackend = () => {
        getFold(this.props.foldId).then((new_fold_data) => {
            console.log(`Got new fold with tags ${new_fold_data.tags}`);
            this.setState({ foldData: new_fold_data });
            if (this.state.foldData?.jobs) {
                // Get current state of jobs as a map
                const currentJobStates = new Map(
                    this.state.jobs?.map(job => [job.id, job]) || []
                );

                // For each job in foldData, determine if we need to refresh it
                const jobsToRefresh = this.state.foldData.jobs
                    .filter(foldJob => {
                        const currentJob = currentJobStates.get(foldJob.id);
                        return foldJob.state === "running" ||
                            currentJob?.state === "running" ||
                            (currentJob && currentJob.state !== foldJob.state);
                    })
                    .map(job => job.id);

                if (jobsToRefresh.length > MAX_JOBS_TO_REFRESH) {
                    notify.warning(`Not streaming job logs because there are too many jobs (${jobsToRefresh.length} > ${MAX_JOBS_TO_REFRESH}))`);
                    return;
                }

                // Create final job list
                const finalJobs = [...this.state.foldData.jobs].map(foldJob => {
                    // Use existing job data if we have it and don't need to refresh
                    if (!jobsToRefresh.includes(foldJob.id)) {
                        return currentJobStates.get(foldJob.id) || foldJob;
                    }
                    return foldJob;
                });

                // Only fetch jobs that need refreshing
                if (jobsToRefresh.length > 0) {
                    Promise.all(
                        jobsToRefresh.map(jobId => getInvokation(jobId))
                    ).then(
                        (refreshedJobs) => {
                            // Update the jobs that were refreshed
                            refreshedJobs.forEach(refreshedJob => {
                                const index = finalJobs.findIndex(j => j.id === refreshedJob.id);
                                if (index !== -1) {
                                    finalJobs[index] = refreshedJob;
                                }
                            });
                            this.setState({ jobs: finalJobs });
                        },
                        (e) => {
                            notify.error(e.toString());
                        }
                    );
                } else {
                    // If no jobs need refreshing, just update state
                    this.setState({ jobs: finalJobs });
                }
            }
        });
    };

    // Generate colors for pfam annotations
    generatePfamColors = (annotations: Annotations): RenderableAnnotations => {
        const colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#FFB347', '#87CEEB', '#F0E68C', '#FA8072',
            '#98FB98', '#F4A460', '#DEB887', '#20B2AA', '#87CEFA'
        ];

        const renderableAnnotations: RenderableAnnotations = {};
        let colorIndex = 0;
        const typeToColor: { [type: string]: string } = {};

        for (const [chainName, chainAnnotations] of Object.entries(annotations)) {
            renderableAnnotations[chainName] = chainAnnotations.map(annotation => {
                // Assign consistent color per annotation type
                if (!typeToColor[annotation.type]) {
                    typeToColor[annotation.type] = colors[colorIndex % colors.length];
                    colorIndex++;
                }

                return {
                    ...annotation,
                    color: typeToColor[annotation.type]
                };
            });
        }

        return renderableAnnotations;
    };

    componentDidMount() {
        this.interval = setInterval(() => {
            if (
                this.state.numRefreshes > REFRESH_STATE_MAX_ITERS &&
                this.interval
            ) {
                clearInterval(this.interval);
            }
            this.setState({
                numRefreshes: this.state.numRefreshes + 1,
            });
            this.refreshFoldDataFromBackend();
        }, REFRESH_STATE_PERIOD);

        // ReactSequenceViewer requires jQuery, and who are we to deny them?
        // @ts-ignore
        window.$ = window.jQuery = jquery;

        // @ts-ignore
        UIkit.util.on(document, "beforeshow", "#paeli", (e: any) =>
            this.setState({ paeIsOnScreen: true })
        );
        // @ts-ignore
        UIkit.util.on(document, "beforehide", "#paeli", (e: any) =>
            this.setState({ paeIsOnScreen: false })
        );
        // @ts-ignore
        UIkit.util.on(document, "beforeshow", "#contactli", (e: any) =>
            this.setState({ contactIsOnScreen: true })
        );
        // @ts-ignore
        UIkit.util.on(document, "beforehide", "#contactli", (e: any) =>
            this.setState({ contactIsOnScreen: false })
        );

        window.addEventListener("resize", this.handleResize);

        getFileList(this.props.foldId).then((files: FileInfo[]) => {
            this.setState({ files: files });
        });

        getFold(this.props.foldId).then(
            (new_fold_data) => {
                this.setState({ foldData: new_fold_data });

                if (this.state.foldData?.jobs) {
                    Promise.all(
                        this.state.foldData.jobs.map((inv) => getInvokation(inv.id))
                    ).then(
                        (fullInvs) => {
                            this.setState({ jobs: fullInvs });
                        },
                        (e) => {
                            notify.error(e.toString());
                        }
                    );
                }

                getFoldPfam(this.props.foldId).then(
                    (pfam) => {
                        console.log("Pfam annotations downloaded:", pfam);
                        if (!this.state.foldData) {
                            return;
                        }
                        const renderableAnnotations = this.generatePfamColors(pfam);
                        console.log("Generated renderable pfam annotations:", renderableAnnotations);
                        this.setState({
                            renderablePfamAnnotations: renderableAnnotations,
                        });
                    },
                    (e) => {
                        console.log("Error downloading pfam annotations:", e.toString());
                    }
                );


                getFile(this.props.foldId, `ranked_0.cif`).then(
                    (fileBlob: Blob) => {
                        const reader = new FileReader();
                        reader.onload = (e) => {
                            try {
                                const fileString = e.target?.result as string;
                                this.setState({ cifString: fileString });
                            } catch (err) {
                                console.error("Error parsing CIF:", err);
                                notify.error(`Failed to parse CIF: ${err}`);
                            }
                        };
                        reader.readAsText(fileBlob);
                    },
                    (e: any) => {
                        console.log(`Fetching PDB because CIF failed: ${e}`);
                        getFile(this.props.foldId, `ranked_0.pdb`).then(
                            (fileBlob: Blob) => {
                                const reader = new FileReader();
                                reader.onload = (e) => {
                                    const fileString = e.target?.result as string;
                                    this.setState({ pdbString: fileString });
                                };
                                reader.readAsText(fileBlob);
                            },
                            (e: any) => {
                                console.log(e);
                                this.setState({ structureFailedToLoad: true });
                            }
                        );
                    }
                );
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    }

    componentWillUnmount() {
        window.removeEventListener("resize", this.handleResize);

        if (this.interval) {
            clearInterval(this.interval);
        }
    }

    setSelectedSubsequence = (selection: Selection | null) => {
        this.setState({
            selectedSubsequence: selection,
        });
    }


    render() {
        var structurePane = (
            <div key="structure" style={{ height: "100%" }}>
                <StructurePane
                    cifString={this.state.cifString ?? null}
                    pdbString={this.state.pdbString ?? null}
                    structureFailedToLoad={this.state.structureFailedToLoad}
                    selection={this.state.selectedSubsequence}
                />
            </div>
        );

        var toolViewHeader = (
            <ul
                className="uk-tab"
                data-uk-tab="connect: #switcher; swiping: false"
                id="tab"
                style={{
                    marginBottom: "0px",
                }}
            >
                <li>
                    <a>Inputs</a>
                </li>
                <li>
                    <a>Logs</a>
                </li>
                <li>
                    <a>Files</a>
                </li>
                {/* TODO(jbr): Figure out why we can't pass displayStructure here... */}
                <li>
                    <a>PAE</a>
                </li>
                {/* <li>
                    <a>Contacts</a>
                </li> */}
                <li>
                    <a>Dock</a>
                </li>
                <li>
                    <a>Naturalness</a>
                </li>
                <li>
                    <a>Embed</a>
                </li>
                <li>
                    <a>Evolve</a>
                </li>
                <li>
                    <a>Actions</a>
                </li>
            </ul>
        );
        var toolViewContentPane = (
            <ul className="uk-switcher uk-margin uk-padding-small" id="switcher">
                <li key="sequenceli">
                    {this.state.foldData ? (
                        <SequenceTab
                            foldId={this.props.foldId}
                            foldName={this.state.foldData?.name}
                            foldTags={this.state.foldData?.tags}
                            foldOwner={this.state.foldData?.owner}
                            foldDiffusionSamples={this.state.foldData?.diffusion_samples}
                            foldCreateDate={this.state.foldData?.create_date}
                            foldPublic={this.state.foldData?.public}
                            yamlConfig={this.state.foldData.yaml_config}
                            sequence={this.state.foldData.sequence}
                            renderablePfamAnnotations={this.state.renderablePfamAnnotations}
                            setPublic={this.setPublic}
                            setFoldName={this.setFoldName}
                            setYamlConfig={this.setYamlConfig}
                            addTag={this.addTag}
                            deleteTag={this.deleteTag}
                            handleTagClick={this.handleTagClick}
                            setSelectedSubsequence={this.setSelectedSubsequence}
                            userType={this.props.userType}
                        ></SequenceTab>
                    ) : null}
                </li>

                <li key="jobsli">
                    <JobsTab jobs={this.state.jobs} />
                </li>

                <li key="filesli">
                    <FileTab
                        foldId={this.props.foldId}
                        foldName={this.state.foldData?.name || null}
                        cifString={this.state.cifString || null}
                        maybeDownloadCif={this.maybeDownloadCif}
                        files={this.state.files}
                    />
                </li>

                <li key="paeli" id="paeli">
                    {this.state.paeIsOnScreen ? (
                        <PaeTab
                            foldId={this.props.foldId}
                            foldSequence={this.state.foldData?.sequence || undefined}
                            yamlConfig={this.state.foldData?.yaml_config || undefined}
                        />
                    ) : null}
                </li>

                {/* <li key="contactli" id="contactli">
                    {this.state.contactIsOnScreen ? (
                        <ContactTab
                            foldId={this.props.foldId}
                            foldSequence={this.state.foldData?.sequence || undefined}
                        ></ContactTab>
                    ) : null}
                </li> */}

                <li key="dock">
                    <DockTab
                        foldId={this.props.foldId}
                        foldName={this.state.foldData?.name || null}
                        foldSequence={this.state.foldData?.sequence || undefined}
                        docks={this.state.foldData ? this.state.foldData.docks : null}
                        jobs={this.state.foldData ? this.state.foldData.jobs : null}
                        displayedLigandNames={[]}  // Object.keys(this.state.displayedDocks)
                        // ranks={Object.fromEntries(
                        //     Object.entries(this.state.displayedDocks).map(([key, value]) => [
                        //         key,
                        //         value.frame + 1,
                        //     ])
                        // )}
                        ranks={{}}
                        displayLigandPose={() => { }}
                        shiftFrame={this.shiftFrame}
                        deleteLigandPose={this.deleteLigandPose}
                    />
                </li>

                <li key="Logitli">
                    <NaturalnessTab
                        foldId={this.props.foldId}
                        foldName={this.state.foldData?.name || null}
                        yamlConfig={this.state.foldData?.yaml_config || null}
                        jobs={this.state.jobs}
                        logits={this.state.foldData?.logits || null}
                        setSelectedSubsequence={this.setSelectedSubsequence}
                        openUpLogsForJob={this.openUpLogsForJob}
                    />
                </li>

                <li key="Embedli">
                    <EmbedTab
                        foldId={this.props.foldId}
                        foldName={this.state.foldData?.name || null}
                        jobs={this.state.jobs}
                        embeddings={this.state.foldData?.embeddings || null}
                        openUpLogsForJob={this.openUpLogsForJob}
                    />
                </li>

                <li key="Evolveli">
                    <EvolveTab
                        foldId={this.props.foldId}
                        yamlConfig={this.state.foldData?.yaml_config || null}
                        jobs={this.state.jobs}
                        files={this.state.files}
                        evolutions={this.state.foldData?.evolutions || null}
                        openUpLogsForJob={this.openUpLogsForJob}
                        setSelectedSubsequence={this.setSelectedSubsequence}
                    />
                </li>

                <li key="actionsli">
                    <form>
                        <Card style={{ marginBottom: '16px' }}>
                            <Title level={4}>Job Management</Title>
                            {[...this.actionToStageName].map((actionAndStageName) => {
                                return (
                                    <div key={actionAndStageName[1]}>
                                        <Button
                                            type="primary"
                                            size="small"
                                            style={{ marginLeft: '8px', marginBottom: '8px' }}
                                            onClick={() => this.startStage(actionAndStageName[1])}
                                        >
                                            {actionAndStageName[0]}
                                        </Button>
                                    </div>
                                );
                            })}
                        </Card>

                        <Card style={{ marginBottom: '16px' }}>
                            <Title level={4}>Fold Management</Title>
                            <div>
                                <Button
                                    size="small"
                                    style={{ marginLeft: '8px', marginBottom: '8px' }}
                                    onClick={this.createSimilarFold}
                                    disabled={this.props.userType === "viewer"}
                                >
                                    Create Similar Fold
                                </Button>
                            </div>
                        </Card>
                    </form>
                </li>
            </ul>
        );

        return (
            <div className="tool-page">
                <h2
                    className="uk-margin-left uk-margin-right uk-text-center" // uk-heading-line
                    style={{
                        marginBottom: "0px",
                        // paddingBottom: "20px",
                    }}
                    id="foldname"
                >
                    <b>{this.state.foldData ? this.state.foldData.name : "Loading..."}</b>
                </h2>
                <div className="uk-flex uk-flex-center uk-flex-wrap">
                    {[...(this.state.foldData?.jobs || [])].map((job: Invokation) => {
                        // If it's (dock, embedding, evolve) and it's not running or queued, don't show it.
                        if (
                            (job.type?.startsWith("dock_") || job.type?.startsWith("embed_") || job.type?.startsWith("evolve_") || job.type?.startsWith("logits_")) &&
                            (job.state !== 'running' && job.state !== 'queued')) {
                            return null;
                        }
                        return (
                            <div key={job.id}>
                                {this.renderBadge(job.type || "misc", job.state, job.starttime)}
                                {/* <br /> */}
                            </div>
                        );
                    })}
                </div>

                <div
                    className="uk-grid uk-margin-top tool-panel-container"
                    data-uk-tab="margin: 0"
                    style={{ margin: "0px" }}
                >
                    <div className="uk-width-1-1 uk-width-1-2@m structure-panel">
                        {structurePane}
                    </div>

                    <div
                        className="uk-width-1-1 uk-width-1-2@m tool-panel"
                        style={{ height: "100%", display: "flex", flexDirection: "column" }}
                    >
                        {toolViewHeader}
                        <div
                            className="tool-panel-contents"
                        // style={{ flexGrow: 1, overflowY: "scroll" }}
                        >
                            {toolViewContentPane}
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    ////////////////////////////////////////////////////////////////////////////////
    // UTILITY FUNCTIONS BELOW.
    ////////////////////////////////////////////////////////////////////////////////

    renderBadge = (
        stageName: string,
        state: string | null | undefined,
        starttime: string | null
    ) => {
        if (!state) {
            return null;
        }

        var jobIsSuspiciouslyLongRunning = false;
        if (starttime) {
            const hoursElapsed =
                (new Date().getTime() - new Date(starttime).getTime()) / 36e5;
            jobIsSuspiciouslyLongRunning = hoursElapsed > 24;
        }

        var badgeColor;
        if (state === "failed") {
            badgeColor = "#f0506e";
        } else if (state === "finished") {
            badgeColor = "#777"; // Too light grey: "#E5E5E5";  // green: "#32d296";
        } else if (state === "deferred") {
            badgeColor = "#999999";
        } else {
            if (jobIsSuspiciouslyLongRunning) {
                badgeColor = "eed202";
            } else {
                badgeColor = "#1C87EF";
            }
        }

        return (
            <span
                className="uk-button-small uk-button-default uk-button-badge uk-margin-small-left"
                style={{ color: badgeColor, borderColor: badgeColor }}
            >
                <span>
                    {stageName}: {state}
                    {jobIsSuspiciouslyLongRunning ? (
                        <div uk-tooltip="This job has been marked running for more than 24 hours. It may have failed. You should restart this stage from the 'Actions' tab below.">
                            ⚠️
                        </div>
                    ) : null}
                </span>
            </span>
        );
    };

    startStage = (stage: string) => {
        queueJob(this.props.foldId, stage, true).then(
            () => {
                notify.info(`Successfully started ${stage}.`);
            },
            (e) => {
                notify.error(e.toString());
            }
        );
    };

    createSimilarFold = () => {
        if (!this.state.foldData) {
            return;
        }

        const copyName = `${this.state.foldData.name} Copy`;
        const yamlConfig = this.state.foldData.yaml_config || '';
        const tags = this.state.foldData.tags || [];

        // Navigate to NewFoldView with prepopulated data
        const params = new URLSearchParams({
            name: encodeURIComponent(copyName),
            yaml: encodeURIComponent(yamlConfig),
            tags: encodeURIComponent(JSON.stringify(tags))
        });

        this.props.navigate(`/newFold?${params.toString()}`);
    };

    maybeDownloadCif = () => {
        if (!this.state.cifString || !this.state.foldData) {
            return;
        }
        fileDownload(this.state.cifString, `${this.state.foldData.name}.cif`);
    };

    actionToStageName = [
        ["Rewrite fasta files", "write_fastas"],
        ["Rerun Sequence Annotation", "annotate"],
        ["Refold", "both"],
    ];

    deleteLigandPose = (ligandId: number, ligandName: string) => {
        UIkit.modal
            .confirm(
                `Are you sure you want to delete the docking result for ${ligandName}?`
            )
            .then(
                () => {
                    // if (ligandName in this.state.displayedDocks) {
                    //     this.state.stage?.removeComponent(
                    //         this.state.displayedDocks[ligandName].nglComponent
                    //     );

                    //     for (const boxComponent of this.state.displayedDocks[ligandName]
                    //         .boxComponents) {
                    //         this.state.stage?.removeComponent(boxComponent);
                    //     }

                    //     const newDisplayedDocks = this.state.displayedDocks;
                    //     delete newDisplayedDocks[ligandName];
                    //     this.setState({ displayedDocks: newDisplayedDocks });
                    //     return;
                    // }

                    deleteDock(ligandId).then(
                        () => {
                            notify.info(`Successfully deleted ligand ${ligandName}.`);
                        },
                        (e) => {
                            UIkit.alert(
                                `Failed to delete dock ${ligandName}... something went wrong.`
                            );
                        }
                    );
                },
                () => { }
            );
    };

    shiftFrame = (ligandName: string, shift: number) => {
        // if (ligandName in this.state.displayedDocks) {
        //     const disp = this.state.displayedDocks[ligandName];
        //     var newFrame = disp.frame + shift;
        //     if (disp.nglComponent.trajList.length) {
        //         if (newFrame < 0) {
        //             newFrame = 0;
        //         }
        //         if (newFrame > disp.nglComponent.structure.frames.length) {
        //             newFrame = disp.nglComponent.structure.frames.length - 1;
        //         }
        //         disp.nglComponent.trajList[0].setFrame(newFrame);
        //     }

        //     const newDisplayedDocks = this.state.displayedDocks;
        //     newDisplayedDocks[ligandName].frame = newFrame;
        //     this.setState({ displayedDocks: newDisplayedDocks });
        // }
    };

    setPublic = (is_public: boolean) => {
        UIkit.modal
            .confirm(
                `Are you sure you want to make this fold and associated data ${is_public ? "" : "in"
                }visible to the public?`
            )
            .then(() => {
                updateFold(this.props.foldId, { public: is_public }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        notify.info("Updated public status.");
                    },
                    (e) => {
                        notify.error(e);
                    }
                );
            });
    };

    setDisableRelaxation = (new_disable_relaxation: boolean) => {
        UIkit.modal
            .confirm(
                `Are you sure you want to set "disable relaxation" to ${new_disable_relaxation
                } for future runs of this fold?`
            )
            .then(() => {
                updateFold(this.props.foldId, { disable_relaxation: new_disable_relaxation }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        notify.info("Updated disable relaxation setting.");
                    },
                    (e) => {
                        notify.error(e);
                    }
                );
            });
    };

    setFoldName = () => {
        UIkit.modal
            .prompt("New fold name:", "")
            .then((newFoldName: string | null) => {
                if (!newFoldName) {
                    return;
                }
                UIkit.modal
                    .confirm(
                        `Are you sure you want to rename this fold to ${newFoldName}?`
                    )
                    .then(() => {
                        updateFold(this.props.foldId, { name: newFoldName }).then(
                            () => {
                                this.refreshFoldDataFromBackend();
                                notify.info("Updated fold name.");
                            },
                            (e) => {
                                notify.error(e);
                            }
                        );
                    });
            });
    };

    setFoldModelPreset = () => {
        UIkit.modal
            .prompt("New fold model:", "")
            .then((newFoldModelPreset: string | null) => {
                if (!newFoldModelPreset) {
                    return;
                }
                updateFold(this.props.foldId, { af2_model_preset: newFoldModelPreset }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        notify.info("Updated fold model.");
                    },
                    (e) => {
                        notify.error(e);
                    }
                );
            });
    };

    setYamlConfig = async (yaml: string) => {
        await updateFold(this.props.foldId, { yaml_config: yaml }).then(
            () => {
                this.refreshFoldDataFromBackend();
            },
            (e) => {
                notify.error(e);
            }
        );
    };

    addTag = (tagToAdd: string) => {
        const tags = this.state.foldData?.tags;
        if (!tags) {
            return;
        }
        tags.push(tagToAdd);
        updateFold(this.props.foldId, { tags: tags }).then(
            () => {
                this.refreshFoldDataFromBackend();
                notify.info("Updated tags.");
            },
            (e) => {
                notify.error(e);
            }
        );
    };

    deleteTag = (tagToDelete: string) => {
        UIkit.modal.confirm("Delete tag?").then(
            () => {
                const tags = this.state.foldData?.tags;
                if (!tags) {
                    return;
                }
                const newTags = tags.filter((tag, index) => tag !== tagToDelete);
                updateFold(this.props.foldId, { tags: newTags }).then(
                    () => {
                        this.refreshFoldDataFromBackend();
                        notify.info("Updated tags.");
                    },
                    (e) => {
                        notify.error(e);
                    }
                );
            },
            () => {
                console.log("Tag deletion cancelled.");
            }
        );
    };

    handleTagClick = (tagToOpen: string) => {
        window.open(`/tag/${tagToOpen}`, "_self");
    };

    formatStartTime = (jobstarttime: string | null) => {
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

    formatRunTime = (jobRunTime: number | null) => {
        return jobRunTime
            ? `${Math.floor(jobRunTime / (60 * 60))} hr ${Math.floor(jobRunTime / 60) % 60
            } min ${Math.floor(jobRunTime) % 60} sec`
            : "NA";
    };
}

function FoldView(props: {
    userType: string | null;
}) {
    let { foldId } = useParams();
    const navigate = useNavigate();

    if (!foldId) {
        return null;
    }
    return (
        <InternalFoldView
            foldId={parseInt(foldId)}
            userType={props.userType}
            navigate={navigate}
        />
    );
}

export default FoldView;
