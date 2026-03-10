/**
 * PCF-MORL: Multi-Objective QoS Optimization for 5G Network Slicing
 * ns-3 5G-LENA Simulation Scenario
 *
 * 1 gNB, 3.5 GHz (n78), 40 MHz, μ=1 (SCS 30 kHz)
 * 2 BWPs on 1 CC: BWP 0 (URLLC 15 MHz) + BWP 1 (eMBB 25 MHz)
 * 14 UEs: 6 URLLC + 8 eMBB
 * Action: Per-slice rate control via TbfQueueDisc
 * Communication: stdin/stdout pipe JSON for Python agent
 */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ideal-beamforming-algorithm.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-helper.h"
#include "ns3/nr-mac-scheduler-tdma-pf.h"
#include "ns3/nr-module.h"
#include "ns3/nr-point-to-point-epc-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/traffic-control-module.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("PcfMorlScenario");

// ============================================================
// Configuration constants
// ============================================================
static const uint32_t NUM_URLLC_UES = 6;
static const uint32_t NUM_EMBB_UES = 8;
static const uint32_t NUM_TOTAL_UES = NUM_URLLC_UES + NUM_EMBB_UES;
static const double CENTRAL_FREQ_HZ = 3.5e9;
static const double TOTAL_BW_HZ = 40e6;
static const double BWP0_BW_HZ = 15e6;
static const double BWP1_BW_HZ = 25e6;
static const uint16_t NUMEROLOGY = 1;
static const double GNB_TX_POWER_DBM = 30.0;
static const double URLLC_DELAY_BUDGET_S = 5e-3;  // 5ms (realistic for NR μ=1)
static const double EMBB_THROUGHPUT_REF_MBPS = 12.0; // per-UE max ~10-12 Mbps on 25MHz BWP

// ============================================================
// KPI collection
// ============================================================
struct SliceKpis
{
    std::vector<double> pdcpDelaysS;
    double rxBytes = 0;
    double txBytes = 0;
    double offeredBytes = 0;
    uint32_t totalUes = 0;

    void Reset()
    {
        pdcpDelaysS.clear();
        rxBytes = 0;
        txBytes = 0;
        offeredBytes = 0;
    }
};

struct SystemKpis
{
    SliceKpis urllc;
    SliceKpis embb;
    double energyJ = 0;

    void Reset()
    {
        urllc.Reset();
        embb.Reset();
        energyJ = 0;
    }
};

// ============================================================
// Globals
// ============================================================
static SystemKpis g_kpis;
static double g_eRef = 1.5e-7;
static uint32_t g_step = 0;
static uint32_t g_maxSteps = 100;
static double g_stepMs = 500.0;
static bool g_done = false;

// Per-UE packet sinks for throughput measurement
static std::vector<Ptr<PacketSink>> g_urllcSinks;
static std::vector<Ptr<PacketSink>> g_embbSinks;

// Previous byte counts for delta computation
static std::vector<uint64_t> g_urllcPrevBytes;
static std::vector<uint64_t> g_embbPrevBytes;

// Application pointers for dynamic rate control
static std::vector<Ptr<UdpClient>> g_urllcPeriodicApps;
static std::vector<Ptr<OnOffApplication>> g_urllcBurstApps;
static std::vector<Ptr<Application>> g_embbApps; // mix of UdpClient and OnOff
static bool g_embbIsCamera[NUM_EMBB_UES]; // true=camera(UdpClient), false=digital twin(OnOff)

// Energy model: load-dependent power
// P = P_idle + P_load_coeff * (offered_bits / capacity_bits)
// Creates QoS vs energy tradeoff: more traffic → better QoS but worse energy
static const double POWER_IDLE_W = 2.0;      // Base power (always on)
static const double POWER_LOAD_W = 8.0;      // Additional power at full load
static const double TOTAL_CAPACITY_BPS = 120e6; // ~120 Mbps total radio capacity

// ============================================================
// Simple JSON helpers (no external dependency)
// ============================================================
static std::string
VecToJson(const std::vector<double>& v)
{
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < v.size(); i++)
    {
        if (i > 0) ss << ",";
        ss << v[i];
    }
    ss << "]";
    return ss.str();
}

static std::string
BuildResponse(const std::vector<double>& obs,
              const std::vector<double>& reward,
              double simTimeS,
              uint32_t step,
              bool done,
              double embbThrMbps,
              double urllcDelay95Ms,
              double urllcVR,
              double energyPerBit)
{
    std::ostringstream ss;
    ss << "{";
    ss << "\"observation\":" << VecToJson(obs) << ",";
    ss << "\"reward\":" << VecToJson(reward) << ",";
    ss << "\"sim_time_s\":" << simTimeS << ",";
    ss << "\"step\":" << step << ",";
    ss << "\"done\":" << (done ? "true" : "false") << ",";
    ss << "\"kpis\":{";
    ss << "\"embb_mean_throughput_mbps\":" << embbThrMbps << ",";
    ss << "\"urllc_delay_95th_ms\":" << urllcDelay95Ms << ",";
    ss << "\"urllc_delay_violation_frac\":" << urllcVR << ",";
    ss << "\"energy_per_bit\":" << energyPerBit;
    ss << "}}";
    return ss.str();
}

static bool
ParseAction(const std::string& line, std::string& action, double& rateUrllc, double& rateEmbb)
{
    // Expected: {"action":"step","rate_urllc_mbps":10,"rate_embb_mbps":50}
    // Simple parser for known format
    action = "";
    rateUrllc = 10;
    rateEmbb = 50;

    auto findVal = [&](const std::string& key) -> std::string {
        size_t pos = line.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = line.find(":", pos);
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '"'))
            pos++;
        size_t end = pos;
        while (end < line.size() && line[end] != ',' && line[end] != '}' && line[end] != '"')
            end++;
        return line.substr(pos, end - pos);
    };

    action = findVal("action");
    std::string ru = findVal("rate_urllc_mbps");
    std::string re = findVal("rate_embb_mbps");
    if (!ru.empty()) rateUrllc = std::stod(ru);
    if (!re.empty()) rateEmbb = std::stod(re);

    return !action.empty();
}

// FlowMonitor for KPI collection
static Ptr<FlowMonitor> g_flowMon;
static Ptr<Ipv4FlowClassifier> g_classifier;

// Port ranges for slice classification
static uint16_t g_urllcPortStart = 0;
static uint16_t g_urllcPortEnd = 0;
static uint16_t g_embbPortStart = 0;
static uint16_t g_embbPortEnd = 0;

// Previous FlowMonitor stats for delta
static std::map<FlowId, FlowMonitor::FlowStats> g_prevStats;

// ============================================================
// Collect KPIs from FlowMonitor + PacketSinks
// ============================================================
static void
CollectFlowKpis()
{
    g_kpis.Reset();
    double stepS = g_stepMs * 1e-3;

    // Collect from PacketSinks (throughput)
    double urllcRx = 0;
    for (size_t i = 0; i < g_urllcSinks.size(); i++)
    {
        uint64_t cur = g_urllcSinks[i]->GetTotalRx();
        urllcRx += (cur - g_urllcPrevBytes[i]);
        g_urllcPrevBytes[i] = cur;
    }
    g_kpis.urllc.rxBytes = urllcRx;

    double embbRx = 0;
    for (size_t i = 0; i < g_embbSinks.size(); i++)
    {
        uint64_t cur = g_embbSinks[i]->GetTotalRx();
        embbRx += (cur - g_embbPrevBytes[i]);
        g_embbPrevBytes[i] = cur;
    }
    g_kpis.embb.rxBytes = embbRx;

    // Collect from FlowMonitor (delay)
    if (g_flowMon && g_classifier)
    {
        auto stats = g_flowMon->GetFlowStats();
        for (auto& [flowId, fs] : stats)
        {
            auto fiveTuple = g_classifier->FindFlow(flowId);
            uint16_t dstPort = fiveTuple.destinationPort;

            // Compute delta stats since last step
            double deltaDelaySum = fs.delaySum.GetSeconds();
            uint64_t deltaRxPackets = fs.rxPackets;
            if (g_prevStats.count(flowId))
            {
                deltaDelaySum -= g_prevStats[flowId].delaySum.GetSeconds();
                deltaRxPackets -= g_prevStats[flowId].rxPackets;
            }

            double avgDelay = (deltaRxPackets > 0) ? deltaDelaySum / deltaRxPackets : 0;

            // Classify by port range
            if (dstPort >= g_urllcPortStart && dstPort < g_urllcPortEnd)
            {
                // Use avg delay as approximation of per-packet delays
                for (uint64_t p = 0; p < deltaRxPackets; p++)
                {
                    g_kpis.urllc.pdcpDelaysS.push_back(avgDelay);
                }
            }
            else if (dstPort >= g_embbPortStart && dstPort < g_embbPortEnd)
            {
                for (uint64_t p = 0; p < deltaRxPackets; p++)
                {
                    g_kpis.embb.pdcpDelaysS.push_back(avgDelay);
                }
            }
        }
        g_prevStats = stats;
    }

    // Energy: load-dependent model (use rxBytes as proxy for radio load)
    double totalRxBits = (g_kpis.urllc.rxBytes + g_kpis.embb.rxBytes) * 8.0;
    double loadFrac = std::min(totalRxBits / (TOTAL_CAPACITY_BPS * stepS), 1.0);
    double powerW = POWER_IDLE_W + POWER_LOAD_W * loadFrac;
    g_kpis.energyJ = powerW * stepS;
}

// ============================================================
// Compute observation (12-dim)
// ============================================================
static std::vector<double>
ComputeObs()
{
    std::vector<double> obs(12, 0.0);
    double stepS = g_stepMs * 1e-3;

    double urllcRx = g_kpis.urllc.rxBytes;
    double embbRx = g_kpis.embb.rxBytes;

    // URLLC slice (0-4)
    double urllcExpPerUe = 2.048e6 / 8.0 * stepS;
    obs[0] = std::min((urllcRx / std::max(1u, NUM_URLLC_UES)) / (7e6 / 8.0 * stepS), 1.0);
    obs[1] = std::min(urllcRx / (NUM_URLLC_UES * urllcExpPerUe), 1.0);

    double urllcDelay95 = 0;
    if (!g_kpis.urllc.pdcpDelaysS.empty())
    {
        auto sorted = g_kpis.urllc.pdcpDelaysS;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min(static_cast<size_t>(0.95 * sorted.size()), sorted.size() - 1);
        urllcDelay95 = sorted[idx];
    }
    obs[2] = std::min(urllcDelay95 / 0.01, 1.0);
    obs[3] = 0; // buffer backlog
    obs[4] = 1.0; // active UE ratio

    // eMBB slice (5-9)
    double embbExpPerUe = 25e6 / 8.0 * stepS;
    obs[5] = std::min((embbRx / std::max(1u, NUM_EMBB_UES)) / embbExpPerUe, 1.0);
    obs[6] = std::min(embbRx / (NUM_EMBB_UES * embbExpPerUe), 1.0);

    double embbDelay95 = 0;
    if (!g_kpis.embb.pdcpDelaysS.empty())
    {
        auto sorted = g_kpis.embb.pdcpDelaysS;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min(static_cast<size_t>(0.95 * sorted.size()), sorted.size() - 1);
        embbDelay95 = sorted[idx];
    }
    obs[7] = std::min(embbDelay95 / 0.1, 1.0);
    obs[8] = 0;
    obs[9] = 1.0;

    // System (10-11)
    double totalCapacity = (BWP0_BW_HZ + BWP1_BW_HZ) * 3.0;
    obs[10] = std::min((urllcRx + embbRx) * 8.0 / stepS / totalCapacity, 1.0);

    double totalBits = (urllcRx + embbRx) * 8.0;
    double epb = (totalBits > 0) ? g_kpis.energyJ / totalBits : g_eRef * 2;
    obs[11] = std::min(epb / (g_eRef * 2.0), 1.0);

    return obs;
}

// ============================================================
// Compute reward (3-dim)
// ============================================================
static std::vector<double>
ComputeReward()
{
    std::vector<double> r(3, 0.0);
    double stepS = g_stepMs * 1e-3;

    // r1: eMBB throughput QoS
    double embbThrMbps = (g_kpis.embb.rxBytes * 8.0) / stepS / 1e6;
    double perUeThr = embbThrMbps / std::max(1u, NUM_EMBB_UES);
    r[0] = std::min(perUeThr / EMBB_THROUGHPUT_REF_MBPS, 1.0);

    // r2: URLLC delay violation rate
    if (!g_kpis.urllc.pdcpDelaysS.empty())
    {
        uint32_t violations = 0;
        for (double d : g_kpis.urllc.pdcpDelaysS)
        {
            if (d > URLLC_DELAY_BUDGET_S) violations++;
        }
        r[1] = -static_cast<double>(violations) / g_kpis.urllc.pdcpDelaysS.size();
    }

    // r3: Energy efficiency
    double totalBits = (g_kpis.urllc.rxBytes + g_kpis.embb.rxBytes) * 8.0;
    if (totalBits > 0 && g_eRef > 0)
    {
        double epb = g_kpis.energyJ / totalBits;
        r[2] = std::max(-epb / g_eRef, -1.0);
    }
    else
    {
        r[2] = -1.0;
    }

    return r;
}

// ============================================================
// Apply rate action by modifying application data rates
// ============================================================
static void
ApplyAction(double rateUrllcMbps, double rateEmbbMbps)
{
    // URLLC: periodic stays at natural rate (2.048 Mbps per UE)
    // Action controls burst rate cap: rate_urllc=5 → minimal burst, =20 → full burst
    double burstRateMbps = std::max(0.1, (rateUrllcMbps - 2.0) * 1.4);
    for (auto& app : g_urllcBurstApps)
    {
        app->SetAttribute("DataRate",
                          DataRateValue(DataRate(std::to_string(
                              static_cast<uint64_t>(burstRateMbps * 1e6)) + "bps")));
    }

    // eMBB: rateEmbbMbps per UE controls offered load
    for (uint32_t i = 0; i < g_embbApps.size(); i++)
    {
        if (g_embbIsCamera[i])
        {
            // Camera (UdpClient): 1250 bytes/pkt
            double intervalUs = (1250.0 * 8.0) / (rateEmbbMbps * 1e6) * 1e6;
            intervalUs = std::max(100.0, intervalUs);
            auto udp = DynamicCast<UdpClient>(g_embbApps[i]);
            if (udp)
            {
                udp->SetAttribute("Interval",
                                  TimeValue(MicroSeconds(static_cast<uint64_t>(intervalUs))));
            }
        }
        else
        {
            // Digital twin (OnOff): adjust DataRate
            // ~62.5% duty cycle → DataRate = target / 0.625
            auto onoff = DynamicCast<OnOffApplication>(g_embbApps[i]);
            if (onoff)
            {
                double onoffRate = rateEmbbMbps / 0.625;
                onoff->SetAttribute("DataRate",
                                    DataRateValue(DataRate(std::to_string(
                                        static_cast<uint64_t>(onoffRate * 1e6)) + "bps")));
            }
        }
    }
}

// ============================================================
// Step handler - runs at each RL decision boundary
// ============================================================
static void
StepHandler()
{
    // Collect KPIs from FlowMonitor + sinks
    CollectFlowKpis();

    // Compute obs & reward
    auto obs = ComputeObs();
    auto reward = ComputeReward();

    bool done = (g_step >= g_maxSteps);
    double stepS = g_stepMs * 1e-3;

    // KPI extras
    double embbThrMbps = (g_kpis.embb.rxBytes * 8.0) / stepS / 1e6 / std::max(1u, NUM_EMBB_UES);
    double urllcDelay95Ms = 0;
    if (!g_kpis.urllc.pdcpDelaysS.empty())
    {
        auto sorted = g_kpis.urllc.pdcpDelaysS;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min(static_cast<size_t>(0.95 * sorted.size()), sorted.size() - 1);
        urllcDelay95Ms = sorted[idx] * 1e3;
    }
    double urllcVR = 0;
    if (!g_kpis.urllc.pdcpDelaysS.empty())
    {
        uint32_t v = 0;
        for (double d : g_kpis.urllc.pdcpDelaysS)
            if (d > URLLC_DELAY_BUDGET_S) v++;
        urllcVR = static_cast<double>(v) / g_kpis.urllc.pdcpDelaysS.size();
    }
    double totalBits = (g_kpis.urllc.rxBytes + g_kpis.embb.rxBytes) * 8.0;
    double epb = (totalBits > 0) ? g_kpis.energyJ / totalBits : 0;

    // Send response to Python via stdout
    std::string resp = BuildResponse(obs, reward, Simulator::Now().GetSeconds(),
                                     g_step, done, embbThrMbps, urllcDelay95Ms, urllcVR, epb);
    std::cout << resp << std::endl;
    std::cout.flush();

    if (done)
    {
        g_done = true;
        Simulator::Stop();
        return;
    }

    // Read next action from Python via stdin
    std::string line;
    if (!std::getline(std::cin, line))
    {
        g_done = true;
        Simulator::Stop();
        return;
    }

    std::string actionType;
    double rateUrllc, rateEmbb;
    if (ParseAction(line, actionType, rateUrllc, rateEmbb))
    {
        if (actionType == "step")
        {
            ApplyAction(rateUrllc, rateEmbb);
            g_step++;
        }
        else if (actionType == "close")
        {
            g_done = true;
            Simulator::Stop();
            return;
        }
    }

    // Schedule next step
    Simulator::Schedule(MilliSeconds(g_stepMs), &StepHandler);
}

// ============================================================
// Main
// ============================================================
int
main(int argc, char* argv[])
{
    uint32_t seed = 0;
    std::string scenario = "training";
    std::string tddPattern = "DL|DL|DL|S|UL|DL|DL|DL|S|UL|";

    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "Random seed", seed);
    cmd.AddValue("scenario", "Scenario type", scenario);
    cmd.AddValue("stepDuration", "Step duration ms", g_stepMs);
    cmd.AddValue("maxSteps", "Max steps per episode", g_maxSteps);
    cmd.AddValue("eRef", "Energy reference", g_eRef);
    cmd.Parse(argc, argv);

    double simTimeSec = g_maxSteps * g_stepMs / 1000.0 + 1.0;

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    // --------------------------------------------------------
    // Nodes
    // --------------------------------------------------------
    NodeContainer gnbNodes;
    gnbNodes.Create(1);

    NodeContainer urllcNodes;
    urllcNodes.Create(NUM_URLLC_UES);

    NodeContainer embbNodes;
    embbNodes.Create(NUM_EMBB_UES);

    NodeContainer allUeNodes;
    allUeNodes.Add(urllcNodes);
    allUeNodes.Add(embbNodes);

    // --------------------------------------------------------
    // Mobility
    // --------------------------------------------------------
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
    gnbPos->Add(Vector(50.0, 25.0, 10.0));
    mobility.SetPositionAllocator(gnbPos);
    mobility.Install(gnbNodes);

    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    // URLLC UEs (close to gNB, 5-15m)
    uePos->Add(Vector(40.0, 20.0, 1.5));
    uePos->Add(Vector(60.0, 20.0, 1.5));
    uePos->Add(Vector(45.0, 30.0, 1.5));
    uePos->Add(Vector(55.0, 30.0, 1.5));
    uePos->Add(Vector(50.0, 15.0, 1.5));
    uePos->Add(Vector(50.0, 35.0, 1.5));
    // eMBB UEs (spread wider, 15-35m)
    uePos->Add(Vector(25.0, 10.0, 2.5));
    uePos->Add(Vector(75.0, 10.0, 2.5));
    uePos->Add(Vector(25.0, 40.0, 2.5));
    uePos->Add(Vector(75.0, 40.0, 2.5));
    uePos->Add(Vector(35.0, 25.0, 1.5));
    uePos->Add(Vector(65.0, 25.0, 1.5));
    uePos->Add(Vector(50.0, 10.0, 1.5));
    uePos->Add(Vector(50.0, 40.0, 1.5));
    mobility.SetPositionAllocator(uePos);
    mobility.Install(allUeNodes);

    // --------------------------------------------------------
    // NR setup
    // --------------------------------------------------------
    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();

    nrHelper->SetBeamformingHelper(beamHelper);
    nrHelper->SetEpcHelper(epcHelper);

    channelHelper->ConfigureFactories("InH-OfficeMixed", "Default", "ThreeGpp");
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));

    // --------------------------------------------------------
    // Spectrum: 1 CC, 2 BWPs
    // --------------------------------------------------------
    CcBwpCreator ccBwpCreator;
    OperationBandInfo band;
    band.m_centralFrequency = CENTRAL_FREQ_HZ;
    band.m_channelBandwidth = TOTAL_BW_HZ;
    band.m_lowerFrequency = band.m_centralFrequency - band.m_channelBandwidth / 2;
    band.m_higherFrequency = band.m_centralFrequency + band.m_channelBandwidth / 2;

    std::unique_ptr<ComponentCarrierInfo> cc0(new ComponentCarrierInfo());
    cc0->m_ccId = 0;
    cc0->m_centralFrequency = CENTRAL_FREQ_HZ;
    cc0->m_channelBandwidth = TOTAL_BW_HZ;
    cc0->m_lowerFrequency = band.m_lowerFrequency;
    cc0->m_higherFrequency = band.m_higherFrequency;

    // BWP 0: URLLC (15 MHz)
    std::unique_ptr<BandwidthPartInfo> bwp0(new BandwidthPartInfo());
    bwp0->m_bwpId = 0;
    bwp0->m_centralFrequency = band.m_lowerFrequency + BWP0_BW_HZ / 2;
    bwp0->m_channelBandwidth = BWP0_BW_HZ;
    bwp0->m_lowerFrequency = bwp0->m_centralFrequency - bwp0->m_channelBandwidth / 2;
    bwp0->m_higherFrequency = bwp0->m_centralFrequency + bwp0->m_channelBandwidth / 2;
    cc0->AddBwp(std::move(bwp0));

    // BWP 1: eMBB (25 MHz)
    std::unique_ptr<BandwidthPartInfo> bwp1(new BandwidthPartInfo());
    bwp1->m_bwpId = 1;
    bwp1->m_centralFrequency = band.m_lowerFrequency + BWP0_BW_HZ + BWP1_BW_HZ / 2;
    bwp1->m_channelBandwidth = BWP1_BW_HZ;
    bwp1->m_lowerFrequency = bwp1->m_centralFrequency - bwp1->m_channelBandwidth / 2;
    bwp1->m_higherFrequency = bwp1->m_centralFrequency + bwp1->m_channelBandwidth / 2;
    cc0->AddBwp(std::move(bwp1));

    band.AddCc(std::move(cc0));
    channelHelper->AssignChannelsToBands({band});

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    // --------------------------------------------------------
    // Scheduler & BWP manager
    // --------------------------------------------------------
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerTdmaPF"));

    // Route URLLC (5QI=1 GBR_CONV_VOICE) to BWP 0
    // Route eMBB (5QI=8 NGBR_VIDEO_TCP_PREMIUM) to BWP 1
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(0));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_PREMIUM", UintegerValue(1));

    beamHelper->SetAttribute("BeamformingMethod",
                             TypeIdValue(DirectPathBeamforming::GetTypeId()));
    epcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    // Antennas
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    // --------------------------------------------------------
    // Install devices
    // --------------------------------------------------------
    NetDeviceContainer gnbDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDev = nrHelper->InstallUeDevice(allUeNodes, allBwps);

    int64_t randomStream = seed * 100 + 1;
    randomStream += nrHelper->AssignStreams(gnbDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueDev, randomStream);

    // PHY per BWP
    double totalPowerW = std::pow(10.0, GNB_TX_POWER_DBM / 10.0) / 1000.0;
    double bwp0Frac = BWP0_BW_HZ / TOTAL_BW_HZ;
    double bwp1Frac = BWP1_BW_HZ / TOTAL_BW_HZ;

    NrHelper::GetGnbPhy(gnbDev.Get(0), 0)->SetAttribute("Numerology", UintegerValue(NUMEROLOGY));
    NrHelper::GetGnbPhy(gnbDev.Get(0), 0)->SetAttribute(
        "TxPower", DoubleValue(10 * std::log10(bwp0Frac * totalPowerW * 1000)));
    NrHelper::GetGnbPhy(gnbDev.Get(0), 0)->SetAttribute("Pattern", StringValue(tddPattern));

    NrHelper::GetGnbPhy(gnbDev.Get(0), 1)->SetAttribute("Numerology", UintegerValue(NUMEROLOGY));
    NrHelper::GetGnbPhy(gnbDev.Get(0), 1)->SetAttribute(
        "TxPower", DoubleValue(10 * std::log10(bwp1Frac * totalPowerW * 1000)));
    NrHelper::GetGnbPhy(gnbDev.Get(0), 1)->SetAttribute("Pattern", StringValue(tddPattern));

    // --------------------------------------------------------
    // Internet stack
    // --------------------------------------------------------
    auto [remoteHost, remoteHostAddr] =
        epcHelper->SetupRemoteHost("10Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;
    internet.Install(allUeNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(ueDev);

    nrHelper->AttachToClosestGnb(ueDev, gnbDev);

    // (Rate control done via application data rate adjustment, not TBF)

    // --------------------------------------------------------
    // Applications
    // --------------------------------------------------------
    uint16_t dlPort = 1234;
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    g_urllcPortStart = dlPort;

    // URLLC traffic
    for (uint32_t i = 0; i < NUM_URLLC_UES; i++)
    {
        // Periodic: 256 bytes/ms = 2.048 Mbps
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        auto sinkApps = sinkHelper.Install(urllcNodes.Get(i));
        serverApps.Add(sinkApps);
        g_urllcSinks.push_back(DynamicCast<PacketSink>(sinkApps.Get(0)));
        g_urllcPrevBytes.push_back(0);

        UdpClientHelper udpClient(ueIpIface.GetAddress(i), dlPort);
        udpClient.SetAttribute("PacketSize", UintegerValue(256));
        udpClient.SetAttribute("Interval", TimeValue(MilliSeconds(1)));
        udpClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        auto periodicApps = udpClient.Install(remoteHost);
        clientApps.Add(periodicApps);
        g_urllcPeriodicApps.push_back(DynamicCast<UdpClient>(periodicApps.Get(0)));

        Ptr<NrQosRule> rule = Create<NrQosRule>();
        NrQosRule::PacketFilter pf;
        pf.localPortStart = dlPort;
        pf.localPortEnd = dlPort;
        rule->Add(pf);
        NrQosFlow qosFlow(NrQosFlow::GBR_CONV_VOICE);
        nrHelper->ActivateDedicatedQosFlow(ueDev.Get(i), qosFlow, rule);
        dlPort++;

        // Burst: OnOff (Poisson λ≈2/s, 7 Mbps peak)
        uint16_t burstPort = dlPort;
        PacketSinkHelper burstSinkHelper("ns3::UdpSocketFactory",
                                         InetSocketAddress(Ipv4Address::GetAny(), burstPort));
        auto burstSinkApps = burstSinkHelper.Install(urllcNodes.Get(i));
        serverApps.Add(burstSinkApps);

        OnOffHelper onOff("ns3::UdpSocketFactory",
                          InetSocketAddress(ueIpIface.GetAddress(i), burstPort));
        onOff.SetAttribute("DataRate", DataRateValue(DataRate("7Mbps")));
        onOff.SetAttribute("PacketSize", UintegerValue(1024));
        onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.05]"));
        onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.45]"));
        auto burstApps = onOff.Install(remoteHost);
        clientApps.Add(burstApps);
        g_urllcBurstApps.push_back(DynamicCast<OnOffApplication>(burstApps.Get(0)));

        Ptr<NrQosRule> burstRule = Create<NrQosRule>();
        NrQosRule::PacketFilter bpf;
        bpf.localPortStart = burstPort;
        bpf.localPortEnd = burstPort;
        burstRule->Add(bpf);
        NrQosFlow burstFlow(NrQosFlow::GBR_CONV_VOICE);
        nrHelper->ActivateDedicatedQosFlow(ueDev.Get(i), burstFlow, burstRule);
        dlPort++;
    }

    g_urllcPortEnd = dlPort;
    g_embbPortStart = dlPort;

    // eMBB traffic
    for (uint32_t i = 0; i < NUM_EMBB_UES; i++)
    {
        uint32_t ueIdx = NUM_URLLC_UES + i;

        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        auto sinkApps = sinkHelper.Install(embbNodes.Get(i));
        serverApps.Add(sinkApps);
        g_embbSinks.push_back(DynamicCast<PacketSink>(sinkApps.Get(0)));
        g_embbPrevBytes.push_back(0);

        if (i < 4)
        {
            // HD camera: CBR ~25 Mbps (controlled by action)
            g_embbIsCamera[i] = true;
            UdpClientHelper udpClient(ueIpIface.GetAddress(ueIdx), dlPort);
            udpClient.SetAttribute("PacketSize", UintegerValue(1250));
            udpClient.SetAttribute("Interval", TimeValue(MicroSeconds(400))); // 25 Mbps initial
            udpClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
            auto camApps = udpClient.Install(remoteHost);
            clientApps.Add(camApps);
            g_embbApps.push_back(camApps.Get(0));
        }
        else
        {
            // Digital twin: OnOff bursty (controlled by action)
            g_embbIsCamera[i] = false;
            OnOffHelper onOff("ns3::UdpSocketFactory",
                              InetSocketAddress(ueIpIface.GetAddress(ueIdx), dlPort));
            onOff.SetAttribute("DataRate", DataRateValue(DataRate("40Mbps")));
            onOff.SetAttribute("PacketSize", UintegerValue(1400));
            onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
            onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.3]"));
            auto dtApps = onOff.Install(remoteHost);
            clientApps.Add(dtApps);
            g_embbApps.push_back(dtApps.Get(0));
        }

        Ptr<NrQosRule> rule = Create<NrQosRule>();
        NrQosRule::PacketFilter pf;
        pf.localPortStart = dlPort;
        pf.localPortEnd = dlPort;
        rule->Add(pf);
        NrQosFlow qosFlow(NrQosFlow::NGBR_VIDEO_TCP_PREMIUM);
        nrHelper->ActivateDedicatedQosFlow(ueDev.Get(ueIdx), qosFlow, rule);
        dlPort++;
    }

    g_embbPortEnd = dlPort;

    // Start apps
    double appStart = 0.2;
    serverApps.Start(Seconds(appStart - 0.1));
    clientApps.Start(Seconds(appStart));
    serverApps.Stop(Seconds(simTimeSec));
    clientApps.Stop(Seconds(simTimeSec - 0.1));

    // --------------------------------------------------------
    // FlowMonitor setup
    // --------------------------------------------------------
    FlowMonitorHelper flowHelper;
    g_flowMon = flowHelper.InstallAll();
    g_classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());

    // --------------------------------------------------------
    // Initial handshake with Python
    // --------------------------------------------------------
    // Wait for reset command
    std::string initLine;
    std::getline(std::cin, initLine);

    // Send initial obs
    std::vector<double> initObs(12, 0.0);
    std::vector<double> initReward(3, 0.0);
    std::string initResp = BuildResponse(initObs, initReward, 0.0, 0, false, 0, 0, 0, 0);
    std::cout << initResp << std::endl;
    std::cout.flush();

    // Wait for first action
    std::string firstLine;
    std::getline(std::cin, firstLine);

    std::string actionType;
    double rateU, rateE;
    ParseAction(firstLine, actionType, rateU, rateE);
    if (actionType == "step")
    {
        ApplyAction(rateU, rateE);
        g_step = 1;
    }

    // Schedule first step
    Simulator::Schedule(MilliSeconds(g_stepMs), &StepHandler);

    // --------------------------------------------------------
    // Run
    // --------------------------------------------------------
    Simulator::Stop(Seconds(simTimeSec));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
