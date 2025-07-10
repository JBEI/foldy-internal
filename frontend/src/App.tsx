import React, { lazy, Suspense, useEffect, useState } from "react";
import { useJwt } from "react-jwt";
import {
    BrowserRouter,
    Route,
    Routes,
    useLocation,
    useNavigate,
    useSearchParams,
} from "react-router-dom";
import "react-tiny-fab/dist/styles.css";
import "./App.scss";

import { GoogleOAuthProvider } from "@react-oauth/google";
import UIkit from "uikit";
import { Layout, Menu, Button as AntButton, Drawer, Spin } from "antd";
import { MenuOutlined, HomeOutlined, InfoCircleOutlined, SettingOutlined, DatabaseOutlined, TagOutlined } from "@ant-design/icons";
import About from "./components/AboutView/About";
import DashboardView from "./components/DashboardView";
import NewBoltzFoldView from "./components/NewFoldView/NewBoltzFoldView";
// import NewFold from "./components/NewFoldView/NewFold2Uniforms";
// import NewFold from "./components/NewFoldView/NewFoldView";
// import NewFold from "./components/NewFoldView/NewFold";
import SudoPage from "./components/SudoPageView/SudoPage";
import {
    authenticationService,
    currentJwtStringSubject,
    DecodedJwt,
    getDescriptionOfUserType,
    isFullDecodedJwt,
    LoginButton,
} from "./services/authentication.service";
import TagView from "./TagView";
import TagsView from "./components/TagsView";
import { FoldingAtTheDisco, FoldyMascot } from "./util/foldyMascot";
import { notify } from "./services/NotificationService";
import { useKeyboardIntercept } from "./util/keyboardInterceptor";

const AvatarFoldView = lazy(() => import("./components/FoldView/FoldView"));

function CheckForErrorQueryString() {
    const location = useLocation();
    const navigate = useNavigate();
    let params = new URLSearchParams(location.search);

    const queryParamErrorText = params.get("error_message");
    if (!queryParamErrorText) {
        return <div></div>;
    }

    notify.error(queryParamErrorText);

    params.delete("error_message");
    navigate({
        pathname: location.pathname,
        search: params.toString(),
    });

    return <div></div>;
}

interface NavLinkProps {
    href: string;
    children: React.ReactNode;
    external?: boolean;
}

function NavLink({ href, children, external = false }: NavLinkProps) {
    const commonStyles = {
        color: "#fff",
        textDecoration: 'none' as const,
        padding: '8px 12px',
        borderRadius: '4px',
        transition: 'background-color 0.2s',
        fontSize: '15px',
    };

    const handleMouseEnter = (e: React.MouseEvent<HTMLAnchorElement>) => {
        (e.target as HTMLElement).style.backgroundColor = 'rgba(255,255,255,0.1)';
    };

    const handleMouseLeave = (e: React.MouseEvent<HTMLAnchorElement>) => {
        (e.target as HTMLElement).style.backgroundColor = 'transparent';
    };

    if (external) {
        return (
            <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                style={commonStyles}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                {children}
            </a>
        );
    }

    return (
        <a
            href={href}
            style={commonStyles}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            {children}
        </a>
    );
}

function RoutedApp({ token, setToken }: {
    token: string | null;
    setToken: React.Dispatch<React.SetStateAction<string | null>>;
}) {
    const { decodedToken, isExpired } = useJwt(token || '');
    let [searchParams, setSearchParams] = useSearchParams();
    const [cartwheelingMascotList, setCartwheelingMascotList] = useState<React.ReactElement[]>([]);
    const [enableDisco, setEnableDisco] = useState(false);
    const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 960);
    const navigate = useNavigate();
    const location = useLocation();

    var fullDecodedToken: DecodedJwt | null = null;
    if (isFullDecodedJwt(decodedToken)) {
        fullDecodedToken = decodedToken;

        const isNewUser = searchParams.get("new_user");
        if (isNewUser) {
            const newSearchParams = new URLSearchParams(searchParams);
            newSearchParams.delete("new_user");
            setSearchParams(newSearchParams);

            UIkit.modal
                .alert(
                    `<div style="text-align: left;">
                        <h3>🎉 Welcome to ${import.meta.env.VITE_INSTITUTION} Foldy!</h3>
                        <p><strong>Your access level:</strong> ${getDescriptionOfUserType(
                        fullDecodedToken.user_claims.type || ""
                    )}</p>

                        <h4>🧬 What is Foldy?</h4>
                        <p>Foldy is a democratized protein folding platform that uses cutting-edge AI models (like Boltz-2x) to predict protein structures with exceptional accuracy for complex scenarios including multimers, small molecule docking, and nucleic acid interactions.</p>

                        <h4>🚀 Get Started:</h4>
                        <ul>
                            <li>Browse existing structures from the <strong>Dashboard</strong></li>
                            <li>Create predictions by clicking <strong>"NEW"</strong> (editors only)</li>
                            <li>Explore the comprehensive analysis tools in each fold</li>
                        </ul>

                        <div style="background-color: #f6ffed; border: 1px solid #b7eb8f; border-radius: 4px; padding: 12px; margin: 16px 0;">
                            <h4 style="color: #389e0d; margin-top: 0;">📚 Citations & Attribution</h4>
                            <p style="margin-bottom: 8px;">If you publish research using this platform, please consider citing the relevant papers to support the developers. This includes both the Foldy platform and underlying methods like Boltz-2x.</p>
                        </div>

                        <p>Visit the <a href="/about">About page</a> for detailed information, FAQs, and complete citation requirements.</p>
                    </div>`
                )
                .then(() => {
                    navigate("/about");
                });
        }
    }

    useKeyboardIntercept('f', () => {
        setCartwheelingMascotList([...cartwheelingMascotList, <FoldyMascot text={""} moveTextAbove={false} isCartwheeling={true} key={cartwheelingMascotList.length} isKanKaning={false} />]);
    });

    useKeyboardIntercept('k', () => {
        setCartwheelingMascotList([...cartwheelingMascotList, <FoldyMascot text={""} moveTextAbove={false} isCartwheeling={true} key={cartwheelingMascotList.length} isKanKaning={true} />]);
    });

    useKeyboardIntercept('d', () => {
        setEnableDisco(!enableDisco);
    });

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 960);
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const renderLoader = () => {
        return (
            <div style={{ textAlign: 'center', padding: '60px 0' }}>
                <Spin size="large" />
            </div>
        );
    };

    const foldyTitle = (
        <span>
            {import.meta.env.VITE_INSTITUTION} Foldy
            <sub>
                <sub>
                    {fullDecodedToken?.user_claims.type === "viewer" ? "View Only" : null}
                    {fullDecodedToken?.user_claims.type === "editor"
                        ? "Edit Access"
                        : null}
                    {fullDecodedToken?.user_claims.type === "admin"
                        ? "Admin Access"
                        : null}
                </sub>
            </sub>
        </span>
    );

    const foldyWelcomeText = `Welcome to ${import.meta.env.VITE_INSTITUTION} Foldy! Login with an ${import.meta.env.VITE_INSTITUTION} account for edit access, or any other account to view public structures.`;

    const menuItems = [
        {
            key: 'dashboard',
            icon: <HomeOutlined />,
            label: 'Dashboard',
            onClick: () => navigate('/')
        },
        {
            key: 'tags',
            icon: <TagOutlined />,
            label: 'Tags',
            onClick: () => navigate('/tags')
        },
        ...(fullDecodedToken?.user_claims.type === "admin" ? [
            {
                key: 'rq',
                icon: <SettingOutlined />,
                label: 'RQ',
                onClick: () => window.open(`${import.meta.env.VITE_BACKEND_URL}/rq/`, '_blank')
            },
            {
                key: 'dbs',
                icon: <DatabaseOutlined />,
                label: 'DBs',
                onClick: () => window.open(`${import.meta.env.VITE_BACKEND_URL}/admin/`, '_blank')
            },
            {
                key: 'sudo',
                icon: <SettingOutlined />,
                label: 'Sudo Page',
                onClick: () => navigate('/sudopage')
            }
        ] : []),
        {
            key: 'about',
            icon: <InfoCircleOutlined />,
            label: 'About',
            onClick: () => navigate('/about')
        }
    ];

    const desktop_navbar = (
        <Layout.Header
            style={{
                background: "linear-gradient(to left, #28a5f5, #1e87f0)",
                padding: '0 24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
            }}
        >
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
                <a
                    href="/"
                    style={{
                        color: "#fff",
                        textDecoration: 'none',
                        fontSize: '20px',
                        whiteSpace: 'nowrap'
                    }}
                >
                    {foldyTitle}
                </a>
                <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
                    <NavLink href="/">Dashboard</NavLink>
                    <NavLink href="/tags">Tags</NavLink>
                    {fullDecodedToken?.user_claims.type === "admin" && (
                        <>
                            <NavLink href={`${import.meta.env.VITE_BACKEND_URL}/rq/`} external>RQ</NavLink>
                            <NavLink href={`${import.meta.env.VITE_BACKEND_URL}/admin/`} external>DBs</NavLink>
                            <NavLink href="/sudopage">Sudo Page</NavLink>
                        </>
                    )}
                    <NavLink href="/about">About</NavLink>
                </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ color: '#fff' }}>
                    <LoginButton
                        decodedToken={fullDecodedToken}
                        setToken={setToken}
                        isExpired={isExpired}
                    />
                </div>
                {fullDecodedToken && !isExpired ? null : (
                    <FoldyMascot text={foldyWelcomeText} moveTextAbove={false} isCartwheeling={false} isKanKaning={false} />
                )}
            </div>
        </Layout.Header>
    );

    const mobile_navbar = (
        <Layout.Header
            style={{
                background: "linear-gradient(to left, #28a5f5, #1e87f0)",
                zIndex: 100,
                position: "fixed",
                top: 0,
                width: "100%",
                padding: '0 16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
            }}
        >
            <a
                href="/"
                style={{
                    color: "#fff",
                    textDecoration: 'none',
                    fontSize: '20px',
                    flex: '1'
                }}
            >
                {foldyTitle}
            </a>

            <AntButton
                type="text"
                icon={<MenuOutlined />}
                onClick={() => setMobileDrawerOpen(true)}
                style={{
                    color: '#fff',
                    border: 'none',
                    padding: '4px 8px',
                    minWidth: '40px',
                    height: '40px'
                }}
            />

            {fullDecodedToken && !isExpired ? null : (
                <FoldyMascot text={foldyWelcomeText} moveTextAbove={true} isCartwheeling={false} isKanKaning={false} />
            )}
        </Layout.Header>
    );

    return (
        <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
            <div style={{ display: isMobile ? 'none' : 'block' }}>{desktop_navbar}</div>
            <div style={{ display: isMobile ? 'block' : 'none', paddingTop: "80px" }}>{mobile_navbar}</div>

            <CheckForErrorQueryString />

            <Drawer
                title={`${import.meta.env.VITE_INSTITUTION} Foldy`}
                placement="right"
                onClose={() => setMobileDrawerOpen(false)}
                open={mobileDrawerOpen}
                styles={{
                    body: { padding: '24px 0' }
                }}
            >
                <div style={{ marginBottom: '24px', padding: '0 24px' }}>
                    <p>
                        {import.meta.env.VITE_INSTITUTION} Foldy is a web app for
                        predicting and using protein structures based on AlphaFold.
                    </p>
                </div>

                <Menu
                    mode="vertical"
                    items={menuItems}
                    onClick={() => setMobileDrawerOpen(false)}
                    style={{ border: 'none' }}
                />

                <div style={{ marginTop: '24px', padding: '0 24px' }}>
                    <LoginButton
                        setToken={setToken}
                        decodedToken={fullDecodedToken}
                        isExpired={isExpired}
                    />
                </div>
            </Drawer>

            <div
                className={location.pathname.startsWith('/fold/') ?
                    "uk-container-expand" :
                    "uk-width-5-6@xl uk-container-center uk-align-center"
                }
                style={{
                    display: "flex",
                    flexDirection: "column",
                    flexGrow: 1,
                    overflow: "hidden",
                    marginTop: "0px",
                    marginBottom: "0px",
                }}
            >
                <Routes>
                    <Route
                        path="/fold/:foldId"
                        element={
                            <Suspense fallback={renderLoader()}>
                                <AvatarFoldView
                                    userType={
                                        fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                    }
                                />
                            </Suspense>
                        }
                    />
                    <Route
                        path="/tag/:tagStringParam"
                        element={<TagView />}
                    />
                    <Route
                        path="/tags"
                        element={<TagsView />}
                    />
                    <Route
                        path="/newFold"
                        element={
                            <NewBoltzFoldView
                                userType={
                                    fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                }
                            />
                        }
                    />
                    <Route
                        path="/sudopage"
                        element={<SudoPage />}
                    />
                    <Route
                        path="/about"
                        element={
                            <About
                                userType={
                                    fullDecodedToken ? fullDecodedToken.user_claims.type : null
                                }
                            />
                        }
                    />
                    <Route
                        path="/"
                        element={
                            <DashboardView
                                decodedToken={fullDecodedToken}
                            />
                        }
                    />
                </Routes>
            </div>
            {cartwheelingMascotList.length > 0 ? cartwheelingMascotList : null}
            <FoldingAtTheDisco enabled={enableDisco} />
        </div>
    );
}


// -----------------------------------------------
// 1) Make a tiny "Bootstrapping" or "InitApp" component
//    that sets the token from the URL, then calls setInitDone.
// -----------------------------------------------
function InitApp({
    onInitDone,
}: {
    onInitDone: (token: string | null) => void;
}) {
    const [searchParams, setSearchParams] = useSearchParams();

    useEffect(() => {
        // Look for the token in the URL
        const jwtString = searchParams.get("access_token");
        if (jwtString) {
            // Remove access_token from the URL
            const newSearchParams = new URLSearchParams(searchParams);
            newSearchParams.delete("access_token");
            setSearchParams(newSearchParams);

            // Persist the token in localStorage
            localStorage.setItem("currentJwtString", jwtString);
            currentJwtStringSubject.next(jwtString);

            // Let parent know initialization is done
            console.log(`IN INITAPP, calling onInitDone with token: ${jwtString}`);
            onInitDone(jwtString);
        } else {
            // No token in URL; see if there's one in localStorage
            const existingToken = localStorage.getItem("currentJwtString");
            console.log(`IN INITAPP, calling onInitDone with existing token: ${existingToken}`);
            onInitDone(existingToken);
        }
    }, [searchParams]);

    // While we parse the URL and store the token, show a spinner/loader
    return (
        <div style={{ textAlign: 'center', padding: '60px 0' }}>
            <Spin size="large" />
        </div>
    );
}

function App() {
    const [token, setToken] = useState<string | null>(null);
    const [initDone, setInitDone] = useState(false);

    console.log(`IN APP, initDone: ${initDone}`);

    function handleInitDone(tokenFromUrl: string | null) {
        console.log(`IN APP, calling HandleInitDone with token: ${tokenFromUrl}`);
        setToken(tokenFromUrl);
        setInitDone(true);
    }

    if (!import.meta.env.VITE_INSTITUTION) {
        console.error("VITE_INSTITUTION is unset.");
    }
    if (!import.meta.env.VITE_BACKEND_URL) {
        console.error("VITE_BACKEND_URL is unset.");
    }
    if (!import.meta.env.VITE_GOOGLE_CLIENT_ID) {
        console.error("VITE_GOOGLE_CLIENT_ID is unset.");
    }
    return (
        <GoogleOAuthProvider
            clientId={import.meta.env.VITE_GOOGLE_CLIENT_ID || ""}
        >
            <BrowserRouter>
                {initDone ? (
                    <RoutedApp token={token} setToken={setToken} />
                ) : (
                    <InitApp onInitDone={handleInitDone} />
                )}
            </BrowserRouter>
        </GoogleOAuthProvider>
    );
}

export default App;
