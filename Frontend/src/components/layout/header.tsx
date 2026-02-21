import { useNavigate } from 'react-router-dom'
import { Logo } from '@/components/ui/logo'
import { SearchBar } from '@/components/ui/search-bar'
import { cn } from '@/lib/utils'

interface HeaderProps {
  onSearch?: (query: string, image?: File) => void
  defaultQuery?: string
  className?: string
}

export function Header({ onSearch, defaultQuery = '', className }: HeaderProps) {
  const navigate = useNavigate()

  const handleSearch = (query: string, image?: File) => {
    if (onSearch) {
      onSearch(query, image)
    } else {
      navigate('/results', { state: { query, image } })
    }
  }

  return (
    <header
      className={cn(
        'sticky top-0 z-40 w-full border-b border-[var(--color-border)]',
        'bg-[var(--color-surface)]/80 backdrop-blur-xl',
        className,
      )}
    >
      <div className="mx-auto flex h-16 max-w-5xl items-center gap-6 px-4 sm:px-6">
        {/* Logo — click to go home */}
        <button
          onClick={() => navigate('/')}
          className="flex-shrink-0"
          aria-label="Home"
        >
          <Logo size="sm" />
        </button>

        {/* Search bar */}
        <div className="flex-1 max-w-xl">
          <SearchBar
            compact
            onSearch={handleSearch}
            defaultValue={defaultQuery}
            placeholder="Search..."
          />
        </div>

      </div>
    </header>
  )
}
