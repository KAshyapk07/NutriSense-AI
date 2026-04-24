import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft } from 'lucide-react'
import { Logo } from '@/components/ui/logo'

const LAST_UPDATED = 'April 24, 2025'

const sections = [
  {
    title: 'Information We Collect',
    body: `We collect the following information when you use NutriSense AI:

• **Account data** — your name, email address, and authentication credentials when you register or sign in with Google.
• **Food & nutrition queries** — text searches, food names, and nutritional questions you submit to get analysis results.
• **Images** — photographs you upload for food recognition. Images are sent to our servers solely for AI analysis and are not stored permanently.
• **Usage preferences** — dietary goals, filters, and settings you configure within the app.
• **Device metadata** — operating system type and app version, used for diagnostics only.

We do not collect your location, contacts, payment information, or any biometric data.`,
  },
  {
    title: 'How We Use Your Information',
    body: `Your data is used exclusively to operate and improve NutriSense AI:

• Authenticating your account and keeping it secure
• Delivering personalised nutrition analysis, meal recommendations, and food comparisons
• Improving our AI models and search accuracy using anonymised, aggregated patterns
• Sending transactional emails (e.g. password reset) — never marketing emails without consent

We do not sell, rent, or share your personal data with third-party advertisers.`,
  },
  {
    title: 'Third-Party Services',
    body: `NutriSense AI uses the following third-party services to function:

• **Firebase (Google)** — account authentication and session management. Governed by Google's Privacy Policy.
• **Microsoft Azure** — our backend API and compute infrastructure. Data is processed in Azure data centres.
• **Neo4j Aura** — our graph database for storing nutritional knowledge. Hosted on managed cloud infrastructure.
• **Groq** — large language model inference for AI-generated nutrition responses. Queries are processed per Groq's data processing agreement; they are not used to train Groq's models.

All third-party providers are contractually bound to protect your data and may not use it for their own purposes.`,
  },
  {
    title: 'Data Retention',
    body: `Account data is retained for as long as your account is active. If you delete your account, your personal information is removed from our systems within 30 days. Anonymised aggregates derived from usage may be retained indefinitely for product improvement.`,
  },
  {
    title: 'Data Security',
    body: `We use industry-standard security measures including HTTPS/TLS encryption for all data in transit, Firebase Authentication for secure credential management, and access controls that limit employee access to production data. No method of transmission over the internet is 100% secure; we cannot guarantee absolute security but we take all reasonable precautions.`,
  },
  {
    title: 'Your Rights',
    body: `You have the right to:

• **Access** — request a copy of the personal data we hold about you
• **Correction** — ask us to update inaccurate data
• **Deletion** — request deletion of your account and associated data
• **Portability** — receive your data in a machine-readable format

To exercise any of these rights, contact us at the email below. We will respond within 30 days.`,
  },
  {
    title: "Children's Privacy",
    body: `NutriSense AI is not directed at children under 13. We do not knowingly collect personal information from children. If you believe a child has provided us with personal data, please contact us and we will delete it promptly.`,
  },
  {
    title: 'Changes to This Policy',
    body: `We may update this Privacy Policy from time to time. When we do, we will update the "Last updated" date at the top of this page. Continued use of NutriSense AI after changes constitutes acceptance of the revised policy.`,
  },
  {
    title: 'Contact',
    body: `If you have questions or concerns about this Privacy Policy, please contact us at:\n\nkashyapk1305@gmail.com`,
  },
]

function formatBody(text: string) {
  return text.split('\n').map((line, i) => {
    const parts = line.split(/(\*\*[^*]+\*\*)/)
    return (
      <span key={i}>
        {parts.map((part, j) =>
          part.startsWith('**') && part.endsWith('**')
            ? <strong key={j} className="text-white/80 font-semibold">{part.slice(2, -2)}</strong>
            : <span key={j}>{part}</span>
        )}
        {i < text.split('\n').length - 1 && <br />}
      </span>
    )
  })
}

export default function PrivacyPage() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-white/[0.06] bg-[#0A0A0A]/90 backdrop-blur-md">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-1.5 text-white/40 hover:text-white/80 transition-colors text-sm font-sans"
          >
            <ArrowLeft size={15} strokeWidth={1.75} />
            Back
          </button>
          <div className="flex-1" />
          <Logo size="sm" alwaysWhite />
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-16">
        {/* Title */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mb-14"
        >
          <p className="text-[10px] font-sans font-semibold uppercase tracking-[0.45em] text-white/25 mb-4">
            Legal
          </p>
          <h1 className="font-serif text-5xl font-bold leading-tight tracking-tight">
            Privacy <span className="font-light italic">Policy</span>
          </h1>
          <p className="mt-4 text-sm text-white/35 font-sans">
            Last updated: {LAST_UPDATED}
          </p>
          <p className="mt-5 text-[13px] text-white/45 font-sans leading-relaxed max-w-xl">
            NutriSense AI ("we", "our", "us") is committed to protecting your privacy.
            This policy explains what data we collect, how we use it, and your rights regarding that data.
          </p>
        </motion.div>

        {/* Sections */}
        <div className="space-y-10">
          {sections.map((section, idx) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05 * idx, ease: [0.22, 1, 0.36, 1] }}
              className="border-t border-white/[0.07] pt-10"
            >
              <h2 className="font-serif text-xl font-semibold text-white/90 mb-4">
                {section.title}
              </h2>
              <p className="text-[13px] text-white/45 font-sans leading-[1.85] whitespace-pre-line">
                {formatBody(section.body)}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Footer note */}
        <div className="mt-16 pt-8 border-t border-white/[0.07]">
          <p className="text-[11px] text-white/15 font-sans">
            NutriSense AI — For educational purposes only. Not intended as medical advice.
          </p>
        </div>
      </div>
    </div>
  )
}
