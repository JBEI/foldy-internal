import React, { FormEvent, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
    authenticationService,
    DecodedJwt,
} from "../services/authentication.service";
import { makeFoldTable } from "../util/foldTable";
import qs from "query-string";
import debounce from "lodash/debounce";
import { getFoldsWithPagination } from "../api/foldApi";
import { Fold } from "src/types/types";
import { notify } from "../services/NotificationService";
import { Input, Button, Space, Divider, Card, Spin, Pagination, Typography } from 'antd';
import { AppstoreOutlined, PlusOutlined } from '@ant-design/icons';

const { Text } = Typography;

const PAGE_SIZE = 25;

const setQueryStringWithoutPageReload = (qsValue: string) => {
    const newurl =
        window.location.protocol +
        "//" +
        window.location.host +
        window.location.pathname +
        qsValue;

    window.history.pushState({ path: newurl }, "", newurl);
};

const setQueryStringValue = (
    key: string,
    value: string,
    queryString = window.location.search
) => {
    const values = qs.parse(queryString);
    const newQsValue = qs.stringify({ ...values, [key]: value });
    setQueryStringWithoutPageReload(`?${newQsValue}`);
};

function getQueryStringValue(
    key: string,
    queryString = window.location.search
): string[] {
    const values = qs.parse(queryString);
    const value: null | string | (string | null)[] = values[key];
    if (typeof value === "string") {
        return [value];
    }
    if (!value) {
        return [];
    }
    return value.map((e) => e || "");
}

function AuthenticatedDashboardView(props: {
    decodedToken: DecodedJwt;
}) {
    const userEmail: string = props.decodedToken.user_claims.email;

    const filterQueryString = getQueryStringValue("filter");
    const [page, rawSetPage] = useState<string[]>(getQueryStringValue("page"));
    const [filter, setFilter] = useState<string[]>(
        filterQueryString.length !== 0 && filterQueryString[0]
            ? filterQueryString
            : [userEmail]
    );
    const [filterFormValue, setFilterFormValue] = useState<string>(filter[0]);
    const [folds, setFolds] = useState<Fold[] | null>(null);
    const [totalCount, setTotalCount] = useState<number>(0);
    const [searchIsStale, setSearchIsStale] = useState<boolean>(false);

    const setPage = useCallback((newValue: number) => {
        rawSetPage([`${newValue}`]);
        setQueryStringValue("page", `${newValue}`);
    }, []);
    // const setFilter = useCallback(
    //   (newValue) => {
    //     setPage(1);
    //     rawSetFilter(newValue);
    //     setQueryStringValue('filter', newValue);
    //   },
    //   [setPage]
    // );

    const pageNum = page.length !== 0 ? parseInt(page[0]) : 1;

    // Note, a few random stack overflows on the internet suggest
    // using useCallback when debouncing, though I don't know why.
    // So if this breaks, maybe consider that:
    // https://stackoverflow.com/questions/61785903/problems-with-debounce-in-useeffect
    const debouncedGetFolds = useCallback(
        debounce((_filter: string | null) => {
            getFoldsWithPagination(_filter, null, pageNum, PAGE_SIZE).then(
                (result) => {
                    setFolds(result.data);
                    setTotalCount(result.pagination.total);
                    setSearchIsStale(false);
                },
                (e) => {
                    notify.error(e.toString());
                }
            );
        }, 300),
        [pageNum]
    );

    useEffect(() => {
        if (authenticationService.currentJwtStringValue) {
            debouncedGetFolds(filter[0]);
        }
    }, [filter, props, debouncedGetFolds]);

    const updateSearchTerm = (e: FormEvent<HTMLFormElement> | null) => {
        if (e) {
            e.preventDefault();
        }
        setPage(1);
        setQueryStringValue("filter", filterFormValue);
        setFilter([filterFormValue]);
    };

    const updateSearchBarText = (newFilterFormValue: string) => {
        setFilterFormValue(newFilterFormValue);
        setSearchIsStale(true);
        // Trigger search automatically with debouncing
        debouncedSearchUpdate(newFilterFormValue);
    };

    const debouncedSearchUpdate = useCallback(
        debounce((searchTerm: string) => {
            setPage(1);
            setQueryStringValue("filter", searchTerm);
            setFilter([searchTerm]);
        }, 300),
        [setPage]
    );

    const searchForNewTerm = (newTerm: string) => {
        setPage(1);
        setFilterFormValue(newTerm);
        setQueryStringValue("filter", newTerm);
        setFilter([newTerm]);
    };

    const refetchData = () => {
        if (authenticationService.currentJwtStringValue) {
            debouncedGetFolds(filter[0]);
        }
    };

    return authenticationService.currentJwtStringValue ? (
        <div style={{ flexGrow: 1, overflowY: "scroll", padding: '24px' }}>
            <Space direction="vertical" size="large" style={{ width: '100%', justifyContent: 'space-between', display: 'flex' }}>
                {/* Header */}
                {/* <Title level={2} style={{ marginBottom: '8px' }}>Protein Folds</Title> */}
                <div style={{ width: '100%', display: 'flex', gap: '16px', alignItems: 'center' }}>
                    {/* Enhanced Search - fills available width */}
                    <Input.Search
                        placeholder="Search folds by name, owner, or tags..."
                        value={filterFormValue}
                        onChange={(e) => updateSearchBarText(e.target.value)}
                        size="large"
                        allowClear
                        enterButton={false}
                        style={{ flex: 1 }}
                    />

                    {/* Action Buttons */}
                    <Button
                        type="default"
                        size="large"
                        icon={<AppstoreOutlined />}
                        onClick={() => searchForNewTerm(" ")}
                    >
                        View All
                    </Button>

                    <Link to={"/newFold"}>
                        <Button
                            type="primary"
                            size="large"
                            icon={<PlusOutlined />}
                        >
                            New Fold
                        </Button>
                    </Link>
                </div>

                {/* <Divider style={{ margin: '16px 0' }} /> */}

                {folds ? (
                    <div
                        key="loadedDiv"
                        style={{ opacity: searchIsStale ? "60%" : "100%" }}
                    >
                        {/* Folds Table */}
                        <Card>
                            {makeFoldTable(folds, {
                                editable: true,
                                userType: props.decodedToken.user_claims.type,
                                onTagsChange: refetchData
                            })}
                        </Card>

                        {/* Total count and pagination */}
                        <div style={{ textAlign: 'center', marginTop: '24px' }}>
                            {totalCount > PAGE_SIZE ? (
                                <Pagination
                                    current={pageNum}
                                    pageSize={PAGE_SIZE}
                                    total={totalCount}
                                    onChange={setPage}
                                    showSizeChanger={false}
                                    showQuickJumper={false}
                                    showTotal={(total, range) => {
                                        const endRange = Math.min(range[1], total);
                                        return `${range[0]}-${endRange} of ${total} folds`;
                                    }}
                                    hideOnSinglePage={true}
                                    size="default"
                                />
                            ) : (
                                <Text type="secondary" style={{ fontSize: '14px' }}>
                                    {totalCount === 1 ? '1 fold' : `${totalCount} folds`}
                                </Text>
                            )}
                        </div>
                    </div>
                ) : (
                    <div style={{ textAlign: 'center', padding: '60px 0' }} key="unloadedDiv">
                        <Spin size="large" tip="Loading folds...">
                            <div style={{ minHeight: '100px' }} />
                        </Spin>
                    </div>
                )}
            </Space>
        </div>
    ) : null;
}

function DashboardView(props: {
    decodedToken: DecodedJwt | null;
}) {
    if (!props.decodedToken) {
        return null;
    }
    return (
        <AuthenticatedDashboardView
            decodedToken={props.decodedToken}
        />
    );
}

export default DashboardView;
